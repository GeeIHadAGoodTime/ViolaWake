# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

Once v1.0 is released, only the latest minor release will receive security patches.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **security@violawake.com** with:

- Description of the vulnerability
- Steps to reproduce
- Impact assessment (what an attacker could do)
- Any suggested fix (optional)

### What to expect

- **Acknowledgment** within 48 hours of your report
- **Initial assessment** within 5 business days
- **Fix or mitigation** timeline communicated after assessment
- **Credit** in the release notes (unless you prefer anonymity)

We will work with you to understand the issue and coordinate disclosure. Please do not disclose publicly until a fix is available. Repository security advisories are published at https://github.com/GeeIHadAGoodTime/ViolaWake/security/advisories once a fix is released.

## Scope

The following are in scope for security reports:

- **SDK code** (`src/violawake_sdk/`) -- vulnerabilities in the Python library
- **Model integrity** -- issues with model download, SHA-256 verification, or supply chain
- **Console backend** (`console/backend/`) -- authentication bypass, injection, unauthorized access
- **Console frontend** (`console/frontend/`) -- XSS, CSRF, credential exposure
- **Dependencies** -- known CVEs in direct dependencies that affect ViolaWake

The following are out of scope:

- Issues in third-party services (ONNX Runtime, PyAudio, etc.) -- report those upstream
- Denial of service via legitimate API usage (rate limiting is a feature request, not a vulnerability)
- Social engineering attacks
- Issues requiring physical access to the device running ViolaWake

## Model Security

ViolaWake models are ONNX files executed via ONNX Runtime. Users should:

- Only load models from trusted sources
- Verify SHA-256 hashes when downloading models (the SDK does this automatically)
- Not load untrusted `.onnx` files -- ONNX Runtime executes model operations which could have side effects in custom operator builds

## Disclosure Policy

We follow coordinated disclosure:

1. Reporter sends vulnerability details to security@violawake.com
2. We confirm receipt and begin investigation
3. We develop and test a fix
4. We release the fix and publish a security advisory
5. Reporter may disclose publicly after the fix is released
