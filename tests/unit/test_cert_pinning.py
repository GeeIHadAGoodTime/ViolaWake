"""Tests for TLS certificate pinning (violawake_sdk.security.cert_pinning).

Tests cover:
  - PinSet construction, matching, and expiry
  - SPKI hash extraction (both cryptography-lib and ASN.1 fallback paths)
  - Pin verification logic (match, mismatch, strict vs soft mode)
  - Subdomain matching
  - Expired pin set handling
  - Dynamic pin bootstrapping
  - SSL context creation
  - Certificate expiry warnings
  - Integration with download_model (mocked network)
  - Python version fallback for get_verified_chain (3.13+ API)
  - TOFU pin caching thread safety
  - Pin format validation edge cases
"""

from __future__ import annotations

import hashlib
import logging
import socket
import ssl
import struct
import sys
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers: build minimal DER certificates for testing
# ---------------------------------------------------------------------------

def _der_length(length: int) -> bytes:
    """Encode a DER length."""
    if length < 0x80:
        return bytes([length])
    elif length < 0x100:
        return bytes([0x81, length])
    elif length < 0x10000:
        return bytes([0x82]) + length.to_bytes(2, "big")
    else:
        return bytes([0x83]) + length.to_bytes(3, "big")


def _der_sequence(contents: bytes) -> bytes:
    """Wrap contents in a DER SEQUENCE."""
    return b"\x30" + _der_length(len(contents)) + contents


def _der_integer(value: int) -> bytes:
    """Encode a DER INTEGER."""
    # Simple encoding for small positive values
    if value < 0x80:
        payload = bytes([value])
    else:
        n_bytes = (value.bit_length() + 8) // 8  # +8 for sign bit
        payload = value.to_bytes(n_bytes, "big")
    return b"\x02" + _der_length(len(payload)) + payload


def _der_bitstring(contents: bytes) -> bytes:
    """Encode a DER BIT STRING (no unused bits)."""
    payload = b"\x00" + contents  # 0 unused bits
    return b"\x03" + _der_length(len(payload)) + payload


def _der_octet_string(contents: bytes) -> bytes:
    """Encode a DER OCTET STRING."""
    return b"\x04" + _der_length(len(contents)) + contents


def _der_explicit_tag(tag_num: int, contents: bytes) -> bytes:
    """Encode a DER explicit context tag."""
    tag_byte = 0xA0 | tag_num
    return bytes([tag_byte]) + _der_length(len(contents)) + contents


def _build_test_spki(key_data: bytes = b"test-public-key-data-1234567890") -> bytes:
    """Build a minimal SubjectPublicKeyInfo DER structure.

    Returns the raw SPKI bytes (a SEQUENCE containing algorithm + key).
    """
    # Algorithm identifier: rsaEncryption OID (simplified)
    algo_oid = b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01"  # rsaEncryption
    algo_null = b"\x05\x00"
    algorithm = _der_sequence(algo_oid + algo_null)

    # Public key as a BIT STRING
    public_key = _der_bitstring(key_data)

    return _der_sequence(algorithm + public_key)


def _build_test_certificate(
    spki: bytes | None = None,
    key_data: bytes = b"test-public-key-data-1234567890",
) -> bytes:
    """Build a minimal X.509 DER certificate with a known SPKI.

    The certificate is structurally valid DER but not cryptographically
    signed — it's only used for SPKI extraction tests.
    """
    if spki is None:
        spki = _build_test_spki(key_data)

    # version [0] EXPLICIT INTEGER 2 (v3)
    version = _der_explicit_tag(0, _der_integer(2))

    # serialNumber
    serial = _der_integer(12345)

    # signature algorithm (sha256WithRSA, simplified)
    sig_algo = _der_sequence(
        b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x0b" + b"\x05\x00"
    )

    # issuer (minimal: single RDN with CN)
    cn_oid = b"\x06\x03\x55\x04\x03"  # id-at-commonName
    cn_value = b"\x0c\x04test"  # UTF8String "test"
    rdn = _der_sequence(cn_oid + cn_value)
    rdn_set = b"\x31" + _der_length(len(rdn)) + rdn
    issuer = _der_sequence(rdn_set)

    # validity (two UTCTime values)
    not_before = b"\x17\x0d250101000000Z"
    not_after = b"\x17\x0d301231235959Z"
    validity = _der_sequence(not_before + not_after)

    # subject (same as issuer for self-signed)
    subject = issuer

    # tbsCertificate
    tbs = _der_sequence(
        version + serial + sig_algo + issuer + validity + subject + spki
    )

    # signatureAlgorithm (same as above)
    outer_sig_algo = sig_algo

    # signature value (dummy)
    sig_value = _der_bitstring(b"\x00" * 64)

    return _der_sequence(tbs + outer_sig_algo + sig_value)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_spki():
    """Return a (spki_bytes, sha256_hex) tuple for testing."""
    spki = _build_test_spki()
    sha = hashlib.sha256(spki).hexdigest()
    return spki, sha


@pytest.fixture
def sample_cert(sample_spki):
    """Return a (cert_der, spki_sha256) tuple for testing."""
    spki, sha = sample_spki
    cert = _build_test_certificate(spki=spki)
    return cert, sha


@pytest.fixture
def _clean_pin_state():
    """Reset global pin state before and after each test."""
    from violawake_sdk.security.cert_pinning import PINNED_HOSTS, _dynamic_pin_cache

    original_hosts = dict(PINNED_HOSTS)
    original_cache = dict(_dynamic_pin_cache)
    yield
    PINNED_HOSTS.clear()
    PINNED_HOSTS.update(original_hosts)
    _dynamic_pin_cache.clear()
    _dynamic_pin_cache.update(original_cache)


# ---------------------------------------------------------------------------
# PinSet tests
# ---------------------------------------------------------------------------

class TestPinSet:
    def test_construction(self):
        from violawake_sdk.security.cert_pinning import PinSet

        ps = PinSet(
            hostname="example.com",
            pins=frozenset(["abc123", "def456"]),
        )
        assert ps.hostname == "example.com"
        assert len(ps.pins) == 2
        assert ps.include_subdomains is True
        assert ps.expires is None
        assert ps.report_uri is None

    def test_matches_exact(self):
        from violawake_sdk.security.cert_pinning import PinSet

        ps = PinSet(hostname="x.com", pins=frozenset(["aabbcc"]))
        assert ps.matches("aabbcc") is True
        assert ps.matches("AABBCC") is True  # case insensitive
        assert ps.matches("112233") is False

    def test_matches_multiple_pins(self):
        from violawake_sdk.security.cert_pinning import PinSet

        ps = PinSet(hostname="x.com", pins=frozenset(["pin1", "pin2", "pin3"]))
        assert ps.matches("pin1") is True
        assert ps.matches("pin2") is True
        assert ps.matches("pin3") is True
        assert ps.matches("pin4") is False

    def test_is_expired_none(self):
        from violawake_sdk.security.cert_pinning import PinSet

        ps = PinSet(hostname="x.com", pins=frozenset(["a"]))
        assert ps.is_expired() is False

    def test_is_expired_future(self):
        from violawake_sdk.security.cert_pinning import PinSet

        future = datetime.now(timezone.utc) + timedelta(days=365)
        ps = PinSet(hostname="x.com", pins=frozenset(["a"]), expires=future)
        assert ps.is_expired() is False

    def test_is_expired_past(self):
        from violawake_sdk.security.cert_pinning import PinSet

        past = datetime.now(timezone.utc) - timedelta(days=1)
        ps = PinSet(hostname="x.com", pins=frozenset(["a"]), expires=past)
        assert ps.is_expired() is True

    def test_frozen(self):
        from violawake_sdk.security.cert_pinning import PinSet

        ps = PinSet(hostname="x.com", pins=frozenset(["a"]))
        with pytest.raises(AttributeError):
            ps.hostname = "y.com"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SPKI extraction tests
# ---------------------------------------------------------------------------

class TestSpkiExtraction:
    def test_asn1_fallback_extracts_correct_hash(self, sample_cert):
        """The ASN.1 fallback parser should extract the same SPKI hash."""
        from violawake_sdk.security.cert_pinning import _extract_spki_hash_asn1_fallback

        cert_der, expected_sha = sample_cert
        actual_sha = _extract_spki_hash_asn1_fallback(cert_der)
        assert actual_sha == expected_sha

    def test_extract_spki_hash_from_der_cert(self, sample_cert):
        """The main extraction function should return the correct hash."""
        from violawake_sdk.security.cert_pinning import (
            _extract_spki_hash_from_der_cert,
            _extract_spki_hash_asn1_fallback,
        )

        cert_der, expected_sha = sample_cert
        # The main function should fall back to ASN.1 for our synthetic certs
        actual_sha = _extract_spki_hash_from_der_cert(cert_der)
        assert actual_sha == expected_sha

    def test_different_keys_produce_different_hashes(self):
        from violawake_sdk.security.cert_pinning import _extract_spki_hash_asn1_fallback

        cert1 = _build_test_certificate(key_data=b"key-aaa-1111111111111111111111")
        cert2 = _build_test_certificate(key_data=b"key-bbb-2222222222222222222222")
        hash1 = _extract_spki_hash_asn1_fallback(cert1)
        hash2 = _extract_spki_hash_asn1_fallback(cert2)
        assert hash1 != hash2

    def test_invalid_der_raises(self):
        from violawake_sdk.security.cert_pinning import _extract_spki_hash_asn1_fallback

        with pytest.raises(ValueError, match="DER parse error"):
            _extract_spki_hash_asn1_fallback(b"\x00\x01\x02")

    def test_truncated_der_raises(self):
        from violawake_sdk.security.cert_pinning import _extract_spki_hash_asn1_fallback

        cert = _build_test_certificate()
        # Truncate midway through
        with pytest.raises(ValueError):
            _extract_spki_hash_asn1_fallback(cert[:20])


# ---------------------------------------------------------------------------
# Pin verification tests
# ---------------------------------------------------------------------------

class TestVerifyCertificatePin:
    def test_no_pins_configured_returns_true(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, verify_certificate_pin

        # Remove all pins
        PINNED_HOSTS.clear()
        cert_der, _ = sample_cert
        assert verify_certificate_pin("unknown-host.com", cert_der) is True

    def test_matching_pin_returns_true(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, verify_certificate_pin

        cert_der, spki_sha = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset([spki_sha]),
        )
        assert verify_certificate_pin("test.com", cert_der) is True

    def test_mismatching_pin_soft_mode(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, verify_certificate_pin

        cert_der, _ = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["0000000000000000000000000000000000000000000000000000000000000000"]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = verify_certificate_pin("test.com", cert_der, strict=False)
            assert result is False
            assert len(w) == 1
            assert "pin verification FAILED" in str(w[0].message)

    def test_mismatching_pin_strict_mode(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import (
            CertPinError,
            PINNED_HOSTS,
            PinSet,
            verify_certificate_pin,
        )

        cert_der, _ = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["0000000000000000000000000000000000000000000000000000000000000000"]),
        )
        with pytest.raises(CertPinError, match="pin verification FAILED"):
            verify_certificate_pin("test.com", cert_der, strict=True)

    def test_backup_pin_matches(self, sample_cert, _clean_pin_state):
        """If the primary pin doesn't match but a backup does, it should pass."""
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, verify_certificate_pin

        cert_der, spki_sha = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset([
                "0000000000000000000000000000000000000000000000000000000000000000",  # primary (wrong)
                spki_sha,  # backup (correct)
            ]),
        )
        assert verify_certificate_pin("test.com", cert_der) is True

    def test_expired_pin_set_skips_verification(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, verify_certificate_pin

        cert_der, _ = sample_cert
        past = datetime.now(timezone.utc) - timedelta(days=1)
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["wrong_pin"]),
            expires=past,
        )
        # Should return True because pin set is expired — skip verification
        assert verify_certificate_pin("test.com", cert_der) is True

    def test_placeholder_pins_trigger_bootstrap(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import (
            PINNED_HOSTS,
            PinSet,
            _dynamic_pin_cache,
            verify_certificate_pin,
        )

        cert_der, spki_sha = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["DYNAMIC_PIN_FETCH_ON_FIRST_USE"]),
        )
        result = verify_certificate_pin("test.com", cert_der)
        assert result is True
        # Dynamic pin should have been cached
        assert "test.com" in _dynamic_pin_cache
        assert _dynamic_pin_cache["test.com"] == spki_sha


# ---------------------------------------------------------------------------
# Subdomain matching tests
# ---------------------------------------------------------------------------

class TestSubdomainMatching:
    def test_exact_match(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, _resolve_pin_set

        PINNED_HOSTS["example.com"] = PinSet(
            hostname="example.com",
            pins=frozenset(["pin1"]),
        )
        result = _resolve_pin_set("example.com")
        assert result is not None
        assert result.hostname == "example.com"

    def test_subdomain_match(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, _resolve_pin_set

        PINNED_HOSTS["example.com"] = PinSet(
            hostname="example.com",
            pins=frozenset(["pin1"]),
            include_subdomains=True,
        )
        result = _resolve_pin_set("sub.example.com")
        assert result is not None
        assert result.hostname == "example.com"

    def test_subdomain_no_match_when_disabled(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, _resolve_pin_set

        PINNED_HOSTS["example.com"] = PinSet(
            hostname="example.com",
            pins=frozenset(["pin1"]),
            include_subdomains=False,
        )
        result = _resolve_pin_set("sub.example.com")
        assert result is None

    def test_deep_subdomain_match(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, _resolve_pin_set

        PINNED_HOSTS["example.com"] = PinSet(
            hostname="example.com",
            pins=frozenset(["pin1"]),
            include_subdomains=True,
        )
        result = _resolve_pin_set("deep.sub.example.com")
        assert result is not None

    def test_no_match(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, _resolve_pin_set

        PINNED_HOSTS["example.com"] = PinSet(
            hostname="example.com",
            pins=frozenset(["pin1"]),
        )
        result = _resolve_pin_set("other.com")
        assert result is None


# ---------------------------------------------------------------------------
# SSL context tests
# ---------------------------------------------------------------------------

class TestSSLContext:
    def test_create_pinned_ssl_context_returns_context(self):
        from violawake_sdk.security.cert_pinning import create_pinned_ssl_context

        ctx = create_pinned_ssl_context("github.com")
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_context_does_not_disable_standard_verification(self):
        from violawake_sdk.security.cert_pinning import create_pinned_ssl_context

        ctx = create_pinned_ssl_context("example.com")
        # Must NOT disable hostname checking or cert verification
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED


# ---------------------------------------------------------------------------
# Certificate expiry warning tests
# ---------------------------------------------------------------------------

class TestCertExpiryWarning:
    def test_no_warning_for_distant_expiry(self, caplog):
        from violawake_sdk.security.cert_pinning import _check_cert_expiry

        future_date = (datetime.now(timezone.utc) + timedelta(days=365)).strftime(
            "%b %d %H:%M:%S %Y GMT"
        )
        peer_cert = {"notAfter": future_date}
        with caplog.at_level(logging.WARNING):
            _check_cert_expiry(peer_cert, "example.com")
        assert "expires" not in caplog.text

    def test_warning_for_near_expiry(self, caplog):
        from violawake_sdk.security.cert_pinning import _check_cert_expiry

        near_date = (datetime.now(timezone.utc) + timedelta(days=10)).strftime(
            "%b %d %H:%M:%S %Y GMT"
        )
        peer_cert = {"notAfter": near_date}
        with caplog.at_level(logging.WARNING):
            _check_cert_expiry(peer_cert, "example.com")
        assert "expires in" in caplog.text

    def test_error_for_expired_cert(self, caplog):
        from violawake_sdk.security.cert_pinning import _check_cert_expiry

        past_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime(
            "%b %d %H:%M:%S %Y GMT"
        )
        peer_cert = {"notAfter": past_date}
        with caplog.at_level(logging.ERROR):
            _check_cert_expiry(peer_cert, "example.com")
        assert "EXPIRED" in caplog.text

    def test_no_crash_on_missing_notafter(self):
        from violawake_sdk.security.cert_pinning import _check_cert_expiry

        _check_cert_expiry({}, "example.com")  # Should not raise
        _check_cert_expiry(None, "example.com")  # Should not raise


# ---------------------------------------------------------------------------
# add_pins runtime registration
# ---------------------------------------------------------------------------

class TestAddPins:
    # Valid 64-char lowercase hex test pins
    _PIN_A = "a" * 64
    _PIN_B = "b" * 64
    _PIN_C = "c" * 64

    def test_add_pins_registers_new_host(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, add_pins

        add_pins("custom-host.io", frozenset([self._PIN_A]))
        assert "custom-host.io" in PINNED_HOSTS
        assert PINNED_HOSTS["custom-host.io"].matches(self._PIN_A)

    def test_add_pins_overwrites_existing(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, add_pins

        add_pins("custom-host.io", frozenset([self._PIN_A]))
        add_pins("custom-host.io", frozenset([self._PIN_B]))
        assert not PINNED_HOSTS["custom-host.io"].matches(self._PIN_A)
        assert PINNED_HOSTS["custom-host.io"].matches(self._PIN_B)

    def test_add_pins_with_expiry(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, add_pins

        future = datetime.now(timezone.utc) + timedelta(days=90)
        add_pins("custom-host.io", frozenset([self._PIN_A]), expires=future)
        assert PINNED_HOSTS["custom-host.io"].expires == future
        assert not PINNED_HOSTS["custom-host.io"].is_expired()

    def test_add_pins_rejects_invalid_format(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("custom-host.io", frozenset(["short"]))

    def test_add_pins_rejects_uppercase(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("custom-host.io", frozenset(["A" * 64]))

    def test_add_pins_rejects_wrong_length(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("custom-host.io", frozenset(["a" * 63]))


# ---------------------------------------------------------------------------
# verify_connection_pin tests (mocked socket)
# ---------------------------------------------------------------------------

class TestVerifyConnectionPin:
    def test_verify_connection_pin_match(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import (
            PINNED_HOSTS,
            PinSet,
            verify_connection_pin,
        )

        cert_der, spki_sha = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset([spki_sha]),
        )

        mock_sock = mock.MagicMock(spec=ssl.SSLSocket)
        mock_sock.getpeercert.side_effect = lambda binary_form=False: (
            cert_der if binary_form else {"notAfter": "Dec 31 23:59:59 2030 GMT"}
        )

        assert verify_connection_pin(mock_sock, "test.com") is True

    def test_verify_connection_pin_no_cert(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import (
            CertPinError,
            PINNED_HOSTS,
            PinSet,
            verify_connection_pin,
        )

        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["somepin"]),
        )

        mock_sock = mock.MagicMock(spec=ssl.SSLSocket)
        mock_sock.getpeercert.return_value = None

        # Soft mode: returns False
        assert verify_connection_pin(mock_sock, "test.com", strict=False) is False

        # Strict mode: raises
        with pytest.raises(CertPinError, match="No peer certificate"):
            verify_connection_pin(mock_sock, "test.com", strict=True)


# ---------------------------------------------------------------------------
# _verify_host_pin integration (mocked network)
# ---------------------------------------------------------------------------

class TestVerifyHostPin:
    def test_verify_host_pin_success(self, sample_cert, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import (
            PINNED_HOSTS,
            PinSet,
            _verify_host_pin,
        )

        cert_der, spki_sha = sample_cert
        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset([spki_sha]),
        )

        mock_tls_sock = mock.MagicMock(spec=ssl.SSLSocket)
        mock_tls_sock.getpeercert.side_effect = lambda binary_form=False: (
            cert_der if binary_form else {"notAfter": "Dec 31 23:59:59 2030 GMT"}
        )
        mock_tls_sock.__enter__ = mock.MagicMock(return_value=mock_tls_sock)
        mock_tls_sock.__exit__ = mock.MagicMock(return_value=False)

        mock_raw_sock = mock.MagicMock(spec=socket.socket)
        mock_raw_sock.__enter__ = mock.MagicMock(return_value=mock_raw_sock)
        mock_raw_sock.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("socket.create_connection", return_value=mock_raw_sock):
            with mock.patch(
                "violawake_sdk.security.cert_pinning.create_pinned_ssl_context"
            ) as mock_ctx:
                mock_ctx.return_value.wrap_socket.return_value = mock_tls_sock
                result = _verify_host_pin("test.com")
                assert result is True

    def test_verify_host_pin_network_error_soft(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, PinSet, _verify_host_pin

        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["somepin"]),
        )

        with mock.patch(
            "socket.create_connection",
            side_effect=OSError("Connection refused"),
        ):
            # Soft mode: returns False with warning
            result = _verify_host_pin("test.com", strict=False)
            assert result is False

    def test_verify_host_pin_network_error_strict(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import (
            CertPinError,
            PINNED_HOSTS,
            PinSet,
            _verify_host_pin,
        )

        PINNED_HOSTS["test.com"] = PinSet(
            hostname="test.com",
            pins=frozenset(["somepin"]),
        )

        with mock.patch(
            "socket.create_connection",
            side_effect=OSError("Connection refused"),
        ):
            with pytest.raises(CertPinError, match="Could not verify"):
                _verify_host_pin("test.com", strict=True)


# ---------------------------------------------------------------------------
# CertPinError tests
# ---------------------------------------------------------------------------

class TestCertPinError:
    def test_is_exception(self):
        from violawake_sdk.security.cert_pinning import CertPinError

        assert issubclass(CertPinError, Exception)

    def test_message(self):
        from violawake_sdk.security.cert_pinning import CertPinError

        err = CertPinError("pin mismatch for example.com")
        assert "pin mismatch" in str(err)


# ---------------------------------------------------------------------------
# Integration: download_model with pin verification (mocked)
# ---------------------------------------------------------------------------

class TestDownloadModelPinIntegration:
    def test_download_model_uses_requests(self, tmp_path, _clean_pin_state):
        """Verify that download_model uses requests.get for downloading."""
        from violawake_sdk import models

        # Create the expected output file so model_path.stat() doesn't fail
        model_file = tmp_path / "viola_mlp_oww.onnx"
        model_file.write_bytes(b"fake-model-data")

        mock_response = mock.MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"fake-model-data"]
        mock_response.raise_for_status.return_value = None

        # Mock stat to return expected size so post-download size validation passes
        fake_stat = mock.MagicMock()
        fake_stat.st_size = models.MODEL_REGISTRY["viola_mlp_oww"].size_bytes

        with mock.patch.object(models, "get_model_dir", return_value=tmp_path):
            with mock.patch("requests.get", return_value=mock_response) as mock_get:
                with mock.patch.object(models, "_verify_sha256"):
                    with mock.patch("pathlib.Path.stat", return_value=fake_stat):
                        models.download_model(
                            "viola_mlp_oww",
                            force=True,
                            verify=False,
                            skip_verify=True,
                        )
                        mock_get.assert_called_once()

    def test_download_model_cached_skips_download(self, tmp_path, _clean_pin_state):
        """Verify that download_model skips download when file is cached."""
        from violawake_sdk import models

        # Create the expected output file so it appears cached
        model_file = tmp_path / "viola_mlp_oww.onnx"
        model_file.write_bytes(b"fake-model-data")

        with mock.patch.object(models, "get_model_dir", return_value=tmp_path):
            with mock.patch("requests.get") as mock_get:
                with mock.patch.object(models, "_verify_sha256"):
                    result = models.download_model(
                        "viola_mlp_oww",
                        force=False,
                        verify=False,
                        skip_verify=True,
                    )
                    # Should NOT have downloaded since file exists and force=False
                    mock_get.assert_not_called()
                    assert result == model_file


# ---------------------------------------------------------------------------
# Default PINNED_HOSTS registry
# ---------------------------------------------------------------------------

class TestDefaultPinnedHosts:
    def test_github_is_pinned(self):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS

        assert "github.com" in PINNED_HOSTS

    def test_github_objects_is_pinned(self):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS

        assert "objects.githubusercontent.com" in PINNED_HOSTS

    def test_huggingface_is_pinned(self):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS

        assert "huggingface.co" in PINNED_HOSTS

    def test_all_pin_sets_have_pins(self):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS

        for hostname, pin_set in PINNED_HOSTS.items():
            assert len(pin_set.pins) > 0, f"No pins for {hostname}"

    def test_all_pin_sets_not_expired(self):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS

        for hostname, pin_set in PINNED_HOSTS.items():
            assert not pin_set.is_expired(), f"Pin set for {hostname} is expired"


# ---------------------------------------------------------------------------
# Python version fallback for get_verified_chain (3.13+ API)
# ---------------------------------------------------------------------------

class TestPythonVersionFallback:
    """Test that fetch_live_spki_pins gracefully handles Python <3.13."""

    def _make_tls_mock(self, cert_der, chain_return=None, chain_error=None):
        """Build a MagicMock for an SSLSocket without spec (so 3.13 attrs work)."""
        mock_tls_sock = mock.MagicMock()
        mock_tls_sock.getpeercert.return_value = cert_der
        if chain_error:
            mock_tls_sock.get_verified_chain.side_effect = chain_error
        elif chain_return is not None:
            mock_tls_sock.get_verified_chain.return_value = chain_return
        mock_tls_sock.__enter__ = mock.MagicMock(return_value=mock_tls_sock)
        mock_tls_sock.__exit__ = mock.MagicMock(return_value=False)
        return mock_tls_sock

    def _make_raw_sock(self):
        mock_raw_sock = mock.MagicMock(spec=socket.socket)
        mock_raw_sock.__enter__ = mock.MagicMock(return_value=mock_raw_sock)
        mock_raw_sock.__exit__ = mock.MagicMock(return_value=False)
        return mock_raw_sock

    def test_falls_back_to_getpeercert_on_old_python(self):
        """On Python <3.13, should use getpeercert(binary_form=True) fallback."""
        from violawake_sdk.security.cert_pinning import fetch_live_spki_pins

        cert_der = _build_test_certificate()
        mock_tls_sock = self._make_tls_mock(cert_der)
        mock_raw_sock = self._make_raw_sock()
        mock_ctx = mock.MagicMock(spec=ssl.SSLContext)
        mock_ctx.wrap_socket.return_value = mock_tls_sock

        with mock.patch("socket.create_connection", return_value=mock_raw_sock):
            with mock.patch("ssl.create_default_context", return_value=mock_ctx):
                with mock.patch(
                    "violawake_sdk.security.cert_pinning.sys"
                ) as mock_sys:
                    mock_sys.version_info = (3, 12, 0, "final", 0)
                    pins = fetch_live_spki_pins("example.com")

        assert len(pins) == 1
        mock_tls_sock.get_verified_chain.assert_not_called()

    def test_uses_get_verified_chain_on_python_313(self):
        """On Python >=3.13, should attempt get_verified_chain first."""
        from violawake_sdk.security.cert_pinning import fetch_live_spki_pins

        cert_der = _build_test_certificate()
        mock_tls_sock = self._make_tls_mock(cert_der, chain_return=[cert_der])
        mock_raw_sock = self._make_raw_sock()
        mock_ctx = mock.MagicMock(spec=ssl.SSLContext)
        mock_ctx.wrap_socket.return_value = mock_tls_sock

        with mock.patch("socket.create_connection", return_value=mock_raw_sock):
            with mock.patch("ssl.create_default_context", return_value=mock_ctx):
                with mock.patch(
                    "violawake_sdk.security.cert_pinning.sys"
                ) as mock_sys:
                    mock_sys.version_info = (3, 13, 0, "final", 0)
                    pins = fetch_live_spki_pins("example.com")

        assert len(pins) == 1
        mock_tls_sock.get_verified_chain.assert_called_once()

    def test_falls_back_when_get_verified_chain_raises_attribute_error(self):
        """Even on 3.13+, if get_verified_chain raises AttributeError, fall back."""
        from violawake_sdk.security.cert_pinning import fetch_live_spki_pins

        cert_der = _build_test_certificate()
        mock_tls_sock = self._make_tls_mock(cert_der, chain_error=AttributeError)
        mock_raw_sock = self._make_raw_sock()
        mock_ctx = mock.MagicMock(spec=ssl.SSLContext)
        mock_ctx.wrap_socket.return_value = mock_tls_sock

        with mock.patch("socket.create_connection", return_value=mock_raw_sock):
            with mock.patch("ssl.create_default_context", return_value=mock_ctx):
                with mock.patch(
                    "violawake_sdk.security.cert_pinning.sys"
                ) as mock_sys:
                    mock_sys.version_info = (3, 13, 0, "final", 0)
                    pins = fetch_live_spki_pins("example.com")

        assert len(pins) == 1


# ---------------------------------------------------------------------------
# TOFU pin caching thread safety
# ---------------------------------------------------------------------------

class TestTofuCacheThreadSafety:
    """Verify that concurrent TOFU bootstrapping is safe."""

    def test_concurrent_bootstrap_same_host(self, _clean_pin_state):
        """Multiple threads bootstrapping the same host should not corrupt the cache."""
        from violawake_sdk.security.cert_pinning import (
            PINNED_HOSTS,
            PinSet,
            _dynamic_pin_cache,
            verify_certificate_pin,
        )

        cert_der, spki_sha = (
            _build_test_certificate(),
            hashlib.sha256(_build_test_spki()).hexdigest(),
        )
        PINNED_HOSTS["race.com"] = PinSet(
            hostname="race.com",
            pins=frozenset(["DYNAMIC_PIN_FETCH_ON_FIRST_USE"]),
        )

        errors = []
        barrier = threading.Barrier(10)

        def worker():
            try:
                barrier.wait(timeout=5)
                result = verify_certificate_pin("race.com", cert_der)
                if not result:
                    errors.append("verify returned False")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        # Cache should have exactly one entry for this host
        assert "race.com" in _dynamic_pin_cache
        assert _dynamic_pin_cache["race.com"] == spki_sha

    def test_concurrent_bootstrap_different_hosts(self, _clean_pin_state):
        """Concurrent bootstrapping of different hosts should not interfere."""
        from violawake_sdk.security.cert_pinning import (
            PINNED_HOSTS,
            PinSet,
            _dynamic_pin_cache,
            verify_certificate_pin,
        )

        certs = {}
        for i in range(5):
            host = f"host{i}.com"
            key_data = f"key-data-for-host-{i}-padding!!!!!!".encode()
            cert = _build_test_certificate(key_data=key_data)
            spki = _build_test_spki(key_data=key_data)
            certs[host] = (cert, hashlib.sha256(spki).hexdigest())
            PINNED_HOSTS[host] = PinSet(
                hostname=host,
                pins=frozenset(["DYNAMIC_PIN_FETCH_ON_FIRST_USE"]),
            )

        errors = []
        barrier = threading.Barrier(5)

        def worker(hostname, cert_der):
            try:
                barrier.wait(timeout=5)
                result = verify_certificate_pin(hostname, cert_der)
                if not result:
                    errors.append(f"verify returned False for {hostname}")
            except Exception as e:
                errors.append(f"{hostname}: {e}")

        threads = [
            threading.Thread(target=worker, args=(h, c))
            for h, (c, _) in certs.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        for host, (_, expected_sha) in certs.items():
            assert _dynamic_pin_cache[host] == expected_sha


# ---------------------------------------------------------------------------
# Pin format validation edge cases
# ---------------------------------------------------------------------------

class TestPinFormatValidationEdgeCases:
    """Edge cases for the add_pins format validator."""

    def test_rejects_empty_string(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("host.com", frozenset([""]))

    def test_rejects_non_hex_chars(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        # 64 chars but contains 'g' which is not hex
        bad_pin = "g" * 64
        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("host.com", frozenset([bad_pin]))

    def test_rejects_mixed_case(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        # Mixed case — validator requires lowercase
        mixed = "aAbBcCdD" * 8  # 64 chars
        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("host.com", frozenset([mixed]))

    def test_rejects_65_chars(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("host.com", frozenset(["a" * 65]))

    def test_rejects_whitespace_padded(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("host.com", frozenset([" " + "a" * 63]))

    def test_accepts_all_hex_digits(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, add_pins

        # Pin using all hex digits 0-9, a-f
        valid_pin = "0123456789abcdef" * 4  # 64 chars
        add_pins("hex-test.com", frozenset([valid_pin]))
        assert PINNED_HOSTS["hex-test.com"].matches(valid_pin)

    def test_rejects_one_bad_pin_in_set(self, _clean_pin_state):
        from violawake_sdk.security.cert_pinning import add_pins

        good_pin = "a" * 64
        bad_pin = "short"
        with pytest.raises(ValueError, match="Invalid pin format"):
            add_pins("host.com", frozenset([good_pin, bad_pin]))

    def test_accepts_empty_frozenset(self, _clean_pin_state):
        """Empty pin set should be accepted (no pins to validate)."""
        from violawake_sdk.security.cert_pinning import PINNED_HOSTS, add_pins

        add_pins("empty-pins.com", frozenset())
        assert "empty-pins.com" in PINNED_HOSTS
        assert len(PINNED_HOSTS["empty-pins.com"].pins) == 0
