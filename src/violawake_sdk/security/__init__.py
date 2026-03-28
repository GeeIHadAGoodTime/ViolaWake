"""
ViolaWake Security — TLS certificate pinning for model downloads.

Provides certificate pinning to prevent MITM attacks during model
distribution. Pins are SHA-256 hashes of the Subject Public Key Info
(SPKI) of the TLS certificates used by download hosts.

Usage::

    from violawake_sdk.security.cert_pinning import create_pinned_ssl_context

    ctx = create_pinned_ssl_context("github.com")
    # Use ctx with requests or urllib
"""

from __future__ import annotations

from violawake_sdk.security.cert_pinning import (
    CertPinError,
    PinSet,
    create_pinned_ssl_context,
    fetch_live_spki_pins,
    verify_certificate_pin,
)

__all__ = [
    "CertPinError",
    "PinSet",
    "create_pinned_ssl_context",
    "fetch_live_spki_pins",
    "verify_certificate_pin",
]
