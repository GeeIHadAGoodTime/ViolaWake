# Security Policy

## Reporting Vulnerabilities
Report security vulnerabilities to security@useviola.com or via GitHub Security Advisories.

## Security Defaults
- Model downloads use HTTPS with SHA-256 integrity verification
- Network audio sources bind to localhost (127.0.0.1) by default
- No pickle serialization — speaker profiles use JSON + numpy .npz
- Certificate pinning infrastructure available (see src/violawake_sdk/security/)

## Model Integrity
Models are verified against SHA-256 hashes in the model registry. If a hash mismatch is detected, the download is rejected and the corrupted file is deleted.
