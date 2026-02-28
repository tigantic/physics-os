# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 40.0.x  | :white_check_mark: |
| 39.x    | :x:                |
| < 39    | :x:                |

## Reporting a Vulnerability

We take the security of The Physics OS seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

📧 **security@tigantic.com**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include

Please include the following information in your report:

- **Type of vulnerability** (e.g., buffer overflow, SQL injection, cross-site scripting)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact assessment** of the vulnerability

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

### Safe Harbor

We consider security research conducted in accordance with this policy to be:

- Authorized concerning any applicable anti-hacking laws
- Authorized concerning any relevant anti-circumvention laws
- Exempt from restrictions in our Terms of Service that would interfere with conducting security research

We will not pursue civil action or initiate a complaint to law enforcement for accidental, good-faith violations of this policy.

### Recognition

We appreciate the security research community's efforts to help keep The Physics OS secure. Researchers who report valid vulnerabilities will be:

- Credited in our security advisories (unless they prefer to remain anonymous)
- Listed in our Security Hall of Fame (for significant findings)

## Security Best Practices for Users

### Dependencies

We recommend:

1. **Pin dependencies** to specific versions in production
2. **Regularly update** to the latest supported version
3. **Monitor** our security advisories

### Cryptographic Considerations

The Ontic Engine uses:

- **Ed25519** for trust certificate signing (server-side only, never exported)
- **Halo2** ZK circuits for zero-knowledge computation integrity proofs
- **Dilithium2** (Post-Quantum Cryptography) for attestation signing (forward-looking)
- **ECDSA** fallback for legacy system interoperability
- **SHA-256** for content-addressed hashing and certificate binding
- **Merkle DAG** for provenance tracking

### Data Handling

- All attestation files are cryptographically signed
- Simulation outputs are content-addressed
- No user data is transmitted externally

## Security-Related Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ONTIC_SIGN_KEY` | Path to signing key | None |
| `ONTIC_VERIFY_SIGS` | Enforce signature verification | `true` |
| `ONTIC_AUDIT_LOG` | Path to audit log | `./audit.log` |

### Audit Logging

Enable comprehensive audit logging:

```bash
export ONTIC_AUDIT_LOG=/var/log/physics_os/audit.log
export ONTIC_AUDIT_LEVEL=INFO
```

## Security Updates

Security updates are released as:

1. **Patch releases** (e.g., 1.5.1) for security fixes
2. **Security advisories** on GitHub
3. **Announcements** on our mailing list

Subscribe to security notifications:
- Watch this repository for "Security Advisories"
- Join the mailing list at security-announce@tigantic.com

---

**Last Updated**: 2026-02-27  
**Policy Version**: 2.1
