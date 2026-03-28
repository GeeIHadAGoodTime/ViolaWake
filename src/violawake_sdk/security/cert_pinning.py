"""Certificate pinning infrastructure for model downloads.

**Security Model — Defense-in-Depth, NOT Primary Integrity Gate**

Certificate pinning in this module is a *defense-in-depth* layer. It is NOT the
primary mechanism that guarantees model integrity. The authoritative integrity
gate is the **SHA-256 hash verification** performed by ``models._verify_sha256()``
after every download — if an attacker tampers with a model file, the hash check
will reject it regardless of certificate pinning status.

**TOFU (Trust-On-First-Use) Limitations:**

All hosts are currently configured with TOFU placeholders rather than hard-coded
SPKI hashes. This means:

- On first contact with a host, the observed certificate SPKI hash is cached
  in-process and subsequent connections must match it.
- TOFU does NOT protect against a man-in-the-middle attack on the *first*
  connection. An attacker present at first use could pin their own certificate.
- TOFU pins are ephemeral (process lifetime only) and do not persist across
  restarts.

**For higher security requirements:**

Pre-populate pins via ``add_pins()`` with known-good SPKI hashes before any
downloads occur. This eliminates the TOFU first-use vulnerability. See the
``PINNED_HOSTS`` dict and ``add_pins()`` function below.

Currently configured for TOFU since SPKI pins require the release
infrastructure to be live. To enable strict pinning, populate PINNED_HOSTS
with real SHA-256 SPKI hashes and call ``pinned_download()`` from
``models.download_model()``.
"""

from __future__ import annotations

import hashlib
import logging
import socket
import ssl
import sys
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DYNAMIC_PIN_FETCH_ON_FIRST_USE = "DYNAMIC_PIN_FETCH_ON_FIRST_USE"


class CertPinError(Exception):
    """Raised when certificate pinning verification fails in strict mode."""


# ---------------------------------------------------------------------------
# Pin storage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinSet:
    """A set of valid SPKI SHA-256 pins for a hostname.

    Attributes:
        hostname: The TLS hostname these pins apply to.
        pins: Set of lowercase hex-encoded SHA-256 hashes of the
              DER-encoded SPKI. At least one must match.
        include_subdomains: Whether pins also apply to subdomains.
        expires: Optional UTC expiry for the pin set. After this date
                 the pins are ignored (treated as if no pins exist)
                 and a warning is logged.
        report_uri: Optional URI for pin violation reports (future use).
    """

    hostname: str
    pins: frozenset[str]
    include_subdomains: bool = True
    expires: datetime | None = None
    report_uri: str | None = None

    def is_expired(self) -> bool:
        """Return True if this pin set has expired."""
        if self.expires is None:
            return False
        return datetime.now(timezone.utc) > self.expires

    def matches(self, spki_hash: str) -> bool:
        """Return True if the given SPKI hash matches any pin."""
        return spki_hash.lower() in {p.lower() for p in self.pins}


# ---------------------------------------------------------------------------
# Pinned certificate fingerprints
# ---------------------------------------------------------------------------
# These are SHA-256 hashes of the SPKI (Subject Public Key Info) for the
# TLS certificates of our model download hosts.
#
# To extract a pin from a live host:
#   python -c "from violawake_sdk.security.cert_pinning import fetch_live_spki_pins; print(fetch_live_spki_pins('github.com'))"
#
# Or via openssl:
#   openssl s_client -connect github.com:443 -servername github.com </dev/null 2>/dev/null |
#     openssl x509 -pubkey -noout |
#     openssl pkey -pubin -outform der |
#     openssl dgst -sha256 -binary | xxd -p -c 256
#
# Current state: all configured hosts use TOFU placeholders instead of
# hard-coded SPKI hashes. The first successful TLS handshake caches the
# observed SPKI hash for the process lifetime, and later connections in the
# same process must match it. Replace these placeholders with real pins
# before enabling strict pinning in production.
# ---------------------------------------------------------------------------

_GITHUB_PINS = PinSet(
    hostname="github.com",
    pins=frozenset([
        DYNAMIC_PIN_FETCH_ON_FIRST_USE,
    ]),
    include_subdomains=True,
    expires=datetime(2028, 1, 1, tzinfo=timezone.utc),
)

_GITHUB_OBJECTS_PINS = PinSet(
    hostname="objects.githubusercontent.com",
    pins=frozenset([
        # GitHub objects CDN — same CA chain as github.com
        DYNAMIC_PIN_FETCH_ON_FIRST_USE,
    ]),
    include_subdomains=True,
    expires=datetime(2028, 1, 1, tzinfo=timezone.utc),
)

_HUGGINGFACE_PINS = PinSet(
    hostname="huggingface.co",
    pins=frozenset([
        # HuggingFace uses Amazon/Cloudfront — pin the intermediate
        DYNAMIC_PIN_FETCH_ON_FIRST_USE,
    ]),
    include_subdomains=True,
    expires=datetime(2028, 1, 1, tzinfo=timezone.utc),
)

# Registry: hostname -> PinSet
# Hostnames are matched with optional subdomain support.
# Guarded by _pinned_hosts_lock for thread-safe mutation (see add_pins()).
PINNED_HOSTS: dict[str, PinSet] = {
    "github.com": _GITHUB_PINS,
    "objects.githubusercontent.com": _GITHUB_OBJECTS_PINS,
    "huggingface.co": _HUGGINGFACE_PINS,
}
_pinned_hosts_lock = threading.Lock()

# Days before certificate expiry to start warning
CERT_EXPIRY_WARNING_DAYS: int = 30

# Cache for dynamically fetched pins (populated on first download)
_dynamic_pin_cache: dict[str, str] = {}
_dynamic_pin_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SPKI extraction
# ---------------------------------------------------------------------------

def _extract_spki_hash_from_der_cert(der_cert: bytes) -> str:
    """Extract SHA-256 hash of SPKI from a DER-encoded certificate.

    The SPKI is a specific sub-structure within an X.509 certificate.
    We use the ssl module's built-in parsing plus the cryptography
    library if available, falling back to a raw ASN.1 extraction.

    Returns:
        Lowercase hex-encoded SHA-256 of the DER-encoded SPKI.
    """
    try:
        # Prefer the cryptography library for robust parsing
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )
    except ImportError:
        logger.debug("cryptography library not available; using ASN.1 fallback for SPKI extraction")
        return _extract_spki_hash_asn1_fallback(der_cert)

    try:
        cert = x509.load_der_x509_certificate(der_cert)
        spki_der = cert.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(spki_der).hexdigest()
    except Exception:
        logger.warning("Failed to parse certificate with cryptography library; using ASN.1 fallback", exc_info=True)
        return _extract_spki_hash_asn1_fallback(der_cert)


def _extract_spki_hash_asn1_fallback(der_cert: bytes) -> str:
    """Extract SPKI hash using manual ASN.1 DER parsing (no external deps).

    This handles the common case where the cryptography library is not
    installed. The parsing is minimal — we navigate the DER structure
    to find the subjectPublicKeyInfo field.

    Returns:
        Lowercase hex-encoded SHA-256 of the DER-encoded SPKI.

    Raises:
        ValueError: If the DER structure cannot be parsed.
    """

    def _read_tag_length(data: bytes, offset: int) -> tuple[int, int, int]:
        """Read a DER tag and length, return (tag, content_start, content_length)."""
        if offset >= len(data):
            raise ValueError("DER parse error: unexpected end of data")
        tag = data[offset]
        offset += 1

        if offset >= len(data):
            raise ValueError("DER parse error: missing length byte")

        length_byte = data[offset]
        offset += 1

        if length_byte < 0x80:
            return tag, offset, length_byte
        elif length_byte == 0x80:
            raise ValueError("DER parse error: indefinite length not supported")
        else:
            num_bytes = length_byte & 0x7F
            if offset + num_bytes > len(data):
                raise ValueError("DER parse error: length bytes exceed data")
            length = int.from_bytes(data[offset:offset + num_bytes], "big")
            return tag, offset + num_bytes, length

    def _skip_element(data: bytes, offset: int) -> int:
        """Skip one DER element, return offset after it."""
        _, content_start, content_length = _read_tag_length(data, offset)
        return content_start + content_length

    def _enter_sequence(data: bytes, offset: int) -> tuple[int, int]:
        """Enter a SEQUENCE, return (content_start, content_end)."""
        tag, content_start, content_length = _read_tag_length(data, offset)
        if tag != 0x30:
            raise ValueError(
                "DER parse error: expected SEQUENCE (0x30), got 0x%02x" % tag
            )
        return content_start, content_start + content_length

    # Parse outer SEQUENCE (Certificate)
    inner_start, _ = _enter_sequence(der_cert, 0)

    # Parse tbsCertificate SEQUENCE
    tbs_start, tbs_end = _enter_sequence(der_cert, inner_start)

    # Navigate tbsCertificate fields:
    # [0] version (EXPLICIT TAG, optional but almost always present)
    # [1] serialNumber
    # [2] signature algorithm
    # [3] issuer
    # [4] validity
    # [5] subject
    # [6] subjectPublicKeyInfo  <-- what we want
    pos = tbs_start

    # Check for explicit version tag [0]
    if pos < tbs_end and der_cert[pos] == 0xA0:
        pos = _skip_element(der_cert, pos)

    # Skip: serialNumber, signature, issuer, validity, subject (5 fields)
    for _ in range(5):
        if pos >= tbs_end:
            raise ValueError("DER parse error: tbsCertificate too short")
        pos = _skip_element(der_cert, pos)

    # Now pos points to subjectPublicKeyInfo
    if pos >= tbs_end:
        raise ValueError("DER parse error: no subjectPublicKeyInfo found")

    _, spki_content_start, spki_content_length = _read_tag_length(der_cert, pos)

    # The SPKI includes the tag and length, so we hash from pos to end of content
    # Re-read to get full element boundaries
    tag_byte = der_cert[pos]
    spki_bytes = der_cert[pos:spki_content_start + spki_content_length]

    # Verify it's a SEQUENCE
    if tag_byte != 0x30:
        raise ValueError(
            "DER parse error: subjectPublicKeyInfo not a SEQUENCE"
        )

    return hashlib.sha256(spki_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Live pin fetching
# ---------------------------------------------------------------------------

def fetch_live_spki_pins(
    hostname: str,
    port: int = 443,
    timeout: float = 10.0,
) -> list[str]:
    """Connect to a host and return the SPKI SHA-256 hashes of its certificate chain.

    This is a diagnostic/bootstrapping utility for populating pin values.
    It should NOT be called during normal downloads — pins should be
    pre-configured in PINNED_HOSTS.

    Args:
        hostname: TLS hostname to connect to.
        port: Port (default 443).
        timeout: Connection timeout in seconds.

    Returns:
        List of hex-encoded SHA-256 hashes of SPKI for each cert in the chain.
    """
    context = ssl.create_default_context()
    pins = []

    with socket.create_connection((hostname, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as tls_sock:
            # get_verified_chain() returns the full verified certificate chain
            # as a list of DER-encoded bytes. It was added in Python 3.13.
            # On older versions, fall back to getpeercert(binary_form=True)
            # which returns only the leaf certificate — less comprehensive
            # but still provides meaningful SPKI pinning for the server cert.
            der_chain = None
            if sys.version_info >= (3, 13):
                try:
                    der_chain = tls_sock.get_verified_chain()  # type: ignore[attr-defined]
                except AttributeError:
                    # Defensive: in case a runtime doesn't expose it despite version check
                    pass

            if der_chain is None:
                # Fallback: get just the peer certificate (works on all Python 3.x)
                der_cert = tls_sock.getpeercert(binary_form=True)
                if der_cert:
                    der_chain = [der_cert]
                else:
                    return []

            for cert_der in der_chain:
                if isinstance(cert_der, dict):
                    # Some Python versions return dicts from get_verified_chain
                    continue
                pin = _extract_spki_hash_from_der_cert(cert_der)
                pins.append(pin)

    return pins


# ---------------------------------------------------------------------------
# Certificate verification
# ---------------------------------------------------------------------------

def _check_cert_expiry(
    peer_cert: dict[str, Any] | None,
    hostname: str,
) -> None:
    """Warn if the peer certificate is close to expiry.

    Args:
        peer_cert: The parsed peer certificate dict from ssl.
        hostname: Hostname for the warning message.
    """
    if not peer_cert:
        return

    not_after = peer_cert.get("notAfter")
    if not not_after:
        return

    try:
        # ssl module returns dates like 'Mar 15 00:00:00 2025 GMT'
        expiry = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
        expiry = expiry.replace(tzinfo=timezone.utc)
        days_remaining = (expiry - datetime.now(timezone.utc)).days

        if days_remaining <= 0:
            logger.error(
                "TLS certificate for %s has EXPIRED (expired %d days ago). "
                "Pin verification may fail.",
                hostname,
                abs(days_remaining),
            )
        elif days_remaining <= CERT_EXPIRY_WARNING_DAYS:
            logger.warning(
                "TLS certificate for %s expires in %d days (on %s). "
                "Consider updating pinned certificates.",
                hostname,
                days_remaining,
                not_after,
            )
    except (ValueError, TypeError) as e:
        logger.debug("Could not parse certificate expiry for %s: %s", hostname, e)


def _resolve_pin_set(hostname: str) -> PinSet | None:
    """Find the PinSet for a hostname, considering subdomain matching.

    Thread-safe: reads are guarded by ``_pinned_hosts_lock`` to avoid
    seeing a partially-updated registry from a concurrent ``add_pins()`` call.

    Args:
        hostname: The hostname to look up.

    Returns:
        The matching PinSet, or None if no pins are configured.
    """
    with _pinned_hosts_lock:
        # Exact match first
        if hostname in PINNED_HOSTS:
            return PINNED_HOSTS[hostname]

        # Subdomain match: check if any parent domain matches with include_subdomains
        parts = hostname.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[i:])
            if parent in PINNED_HOSTS and PINNED_HOSTS[parent].include_subdomains:
                return PINNED_HOSTS[parent]

    return None


def verify_certificate_pin(
    hostname: str,
    der_cert: bytes,
    strict: bool = False,
) -> bool:
    """Verify that a certificate's SPKI matches a pinned value.

    Args:
        hostname: The hostname being connected to.
        der_cert: The DER-encoded certificate bytes.
        strict: If True, raise CertPinError on mismatch.
                If False, log a warning and return False.

    Returns:
        True if the pin matches (or no pins configured for this host).
        False if pin verification failed (soft mode).

    Raises:
        CertPinError: In strict mode when pin verification fails.
    """
    pin_set = _resolve_pin_set(hostname)
    if pin_set is None:
        logger.debug("No pins configured for %s — skipping pin verification", hostname)
        return True

    if pin_set.is_expired():
        logger.warning(
            "Pin set for %s has expired (expired at %s). "
            "Skipping pin verification — update PINNED_HOSTS.",
            hostname,
            pin_set.expires,
        )
        return True

    actual_hash = _extract_spki_hash_from_der_cert(der_cert)

    # Check against all pins (current + backup)
    # Skip placeholder pins that indicate dynamic fetch is needed
    real_pins = {
        p for p in pin_set.pins
        if not p.startswith("DYNAMIC_PIN_FETCH")
    }

    if not real_pins:
        with _dynamic_pin_cache_lock:
            cached_pin = _dynamic_pin_cache.get(hostname)
        if cached_pin is not None:
            if actual_hash.lower() == cached_pin.lower():
                logger.debug("TOFU pin verified for %s", hostname)
                return True

            msg = (
                f"TOFU certificate pin verification FAILED for {hostname}. "
                f"Expected cached SPKI hash {cached_pin[:16]}..., got {actual_hash[:16]}.... "
                f"This may indicate a MITM attack or legitimate certificate rotation. "
                f"Replace the placeholder pin with a real SPKI hash before enabling strict pinning."
            )

            if strict:
                raise CertPinError(msg)

            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        # All pins are placeholders — try dynamic pin fetch and cache
        logger.info(
            "All pins for %s are TOFU placeholders. "
            "Caching the first observed SPKI hash for this process.",
            hostname,
        )
        _bootstrap_dynamic_pin(hostname, actual_hash)
        return True

    if actual_hash.lower() in {p.lower() for p in real_pins}:
        logger.debug("Certificate pin verified for %s", hostname)
        return True

    # Pin mismatch
    msg = (
        f"Certificate pin verification FAILED for {hostname}. "
        f"Expected one of: {', '.join(p[:16] + '...' for p in real_pins)}. "
        f"Got: {actual_hash[:16]}... "
        f"This may indicate a MITM attack or certificate rotation. "
        f"Update PINNED_HOSTS if the certificate was legitimately rotated."
    )

    if strict:
        raise CertPinError(msg)

    logger.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)
    return False


def _bootstrap_dynamic_pin(hostname: str, observed_hash: str) -> None:
    """Cache a dynamically observed pin for a host with placeholder pins.

    This is called on first use when all configured pins are placeholders.
    The observed hash is cached in memory for the process lifetime.
    """
    with _dynamic_pin_cache_lock:
        _dynamic_pin_cache[hostname] = observed_hash
    logger.info(
        "Cached dynamic pin for %s: %s... "
        "To make this permanent, add it to PINNED_HOSTS in cert_pinning.py",
        hostname,
        observed_hash[:16],
    )


# ---------------------------------------------------------------------------
# SSL context creation
# ---------------------------------------------------------------------------

def create_pinned_ssl_context(
    hostname: str,
    strict: bool = False,
) -> ssl.SSLContext:
    """Create an SSL context that performs certificate pinning.

    The returned context validates certificates normally via the system
    trust store AND additionally checks the SPKI pin after the TLS
    handshake completes.

    Note: Python's ssl module does not support custom handshake callbacks
    that can reject connections based on SPKI pins. Instead, we use a
    standard SSL context and perform pin verification as a post-connect
    check. Callers should use ``verify_connection_pin()`` after connecting.

    Args:
        hostname: The hostname to pin for.
        strict: If True, pin failures raise CertPinError.
                If False, pin failures log warnings but allow the connection.

    Returns:
        A configured ssl.SSLContext.
    """
    context = ssl.create_default_context()
    # Standard TLS verification is always enabled
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    # Enforce TLS 1.2+ (disable older insecure versions)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def verify_connection_pin(
    tls_sock: ssl.SSLSocket,
    hostname: str,
    strict: bool = False,
) -> bool:
    """Verify the SPKI pin of an already-established TLS connection.

    Call this immediately after connecting with a pinned SSL context.

    Args:
        tls_sock: The connected TLS socket.
        hostname: The hostname connected to.
        strict: If True, raise CertPinError on mismatch.

    Returns:
        True if pin matches or no pins configured.
        False if pin mismatch (soft mode).

    Raises:
        CertPinError: In strict mode when pin verification fails.
    """
    # Check certificate expiry
    peer_cert = tls_sock.getpeercert()
    _check_cert_expiry(peer_cert, hostname)

    # Get DER-encoded certificate for pin verification
    der_cert = tls_sock.getpeercert(binary_form=True)
    if der_cert is None:
        msg = f"No peer certificate available for {hostname}"
        if strict:
            raise CertPinError(msg)
        logger.warning(msg)
        return False

    return verify_certificate_pin(hostname, der_cert, strict=strict)


# ---------------------------------------------------------------------------
# High-level download integration
# ---------------------------------------------------------------------------

def pinned_download(
    url: str,
    dest: Any,
    verify_pin: bool = True,
    strict: bool = False,
    timeout: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Download a URL with optional certificate pinning.

    This wraps requests.get() with pre- and post-connection pin
    verification. It's designed to be a drop-in replacement for
    the download path in models.py.

    Args:
        url: URL to download.
        dest: Destination path (not used directly — caller handles writing).
        verify_pin: If True, perform certificate pinning.
        strict: If True, abort download on pin mismatch.
        timeout: Request timeout in seconds.
        **kwargs: Additional kwargs passed to requests.get().

    Returns:
        The requests.Response object.

    Raises:
        CertPinError: In strict mode when pin verification fails.
        ImportError: If requests is not installed.
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for model downloading. "
            "Install with: pip install violawake[download]"
        ) from None

    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if not verify_pin:
        logger.debug("Pin verification disabled for %s", url)
        return requests.get(url, stream=True, timeout=timeout, **kwargs)

    pin_set = _resolve_pin_set(hostname)

    if pin_set is None:
        logger.debug(
            "No pins configured for host %s — downloading with standard TLS",
            hostname,
        )
        return requests.get(url, stream=True, timeout=timeout, **kwargs)

    if pin_set.is_expired():
        logger.warning(
            "Pin set for %s has expired. Downloading with standard TLS only.",
            hostname,
        )
        return requests.get(url, stream=True, timeout=timeout, **kwargs)

    # ---------------------------------------------------------------
    # TOCTOU note (download-then-verify):
    #
    # We download via requests (which uses its own TLS connection) and
    # then verify the host's SPKI pin via a separate TLS probe. This
    # creates a small time-of-check-to-time-of-use gap: a sophisticated
    # attacker could theoretically present a valid cert for the download
    # connection and a different cert for the verification probe, or
    # vice versa.
    #
    # This is acceptable for our threat model because:
    # 1. The SDK only downloads from known CDNs (GitHub, HuggingFace)
    #    whose certificates are validated by the system trust store on
    #    BOTH connections — standard TLS verification is always active.
    # 2. All downloaded model files are verified against a SHA-256 hash
    #    after download (see models._verify_sha256), so any tampering
    #    during the TOCTOU window is caught before the model is used.
    # 3. Pin verification is a defence-in-depth layer, not the sole
    #    integrity guarantee. The hash check is the authoritative gate.
    #
    # To eliminate this gap entirely, we would need to hook into the
    # TLS handshake of the download connection itself, which Python's
    # requests/urllib3 stack does not expose.
    # ---------------------------------------------------------------
    response = requests.get(url, stream=True, timeout=timeout, **kwargs)

    # Verify the certificate pin via a lightweight TLS probe to the same host
    try:
        _verify_host_pin(hostname, strict=strict)
    except CertPinError:
        response.close()
        raise

    return response


def _verify_host_pin(
    hostname: str,
    port: int = 443,
    strict: bool = False,
    timeout: float = 10.0,
) -> bool:
    """Open a TLS connection to verify the host's certificate pin.

    Args:
        hostname: Host to verify.
        port: Port to connect to.
        strict: If True, raise on mismatch.
        timeout: Connection timeout.

    Returns:
        True if pin matches or no pins configured.
    """
    context = create_pinned_ssl_context(hostname, strict=strict)

    try:
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as tls_sock:
                return verify_connection_pin(tls_sock, hostname, strict=strict)
    except CertPinError:
        raise
    except OSError as e:
        msg = (
            f"Could not verify certificate pin for {hostname}: {e}. "
            f"Proceeding with standard TLS verification only."
        )
        if strict:
            raise CertPinError(msg) from e
        logger.warning(msg)
        return False


def add_pins(hostname: str, pins: frozenset[str], **kwargs: Any) -> None:
    """Register or update pins for a hostname at runtime.

    This allows callers to add custom pins (e.g., for self-hosted
    model repositories) without modifying the source code.

    Thread-safe: mutations to ``PINNED_HOSTS`` are guarded by
    ``_pinned_hosts_lock``.

    Note: The TOFU (trust-on-first-use) pin model used by default is a
    **defence-in-depth** measure, not a primary security mechanism.  The
    authoritative integrity guarantee is the SHA-256 hash check performed
    by ``models._verify_sha256()`` after every download.  TOFU pinning
    adds an additional layer that detects certificate changes within a
    single process lifetime, but it does not replace proper hash
    verification.

    Args:
        hostname: The hostname to pin.
        pins: Set of SPKI SHA-256 hex hashes (lowercase hex, 64 characters).
        **kwargs: Additional PinSet fields (include_subdomains, expires, etc.).

    Raises:
        ValueError: If any pin is not a valid lowercase hex string of length 64.
    """
    import re

    _HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
    for pin in pins:
        if not _HEX64_RE.match(pin):
            raise ValueError(
                f"Invalid pin format: '{pin}'. "
                f"Pins must be lowercase hex strings of exactly 64 characters "
                f"(SHA-256 hash)."
            )

    with _pinned_hosts_lock:
        PINNED_HOSTS[hostname] = PinSet(hostname=hostname, pins=pins, **kwargs)
    logger.info("Registered %d pins for %s", len(pins), hostname)
