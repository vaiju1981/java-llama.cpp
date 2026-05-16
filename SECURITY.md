# Security Policy

## Supported Versions

Only the most recent release of the `5.x` series receives security fixes. Older major versions are not actively maintained.

| Version | Supported |
|---------|-----------|
| 5.x (latest) | Yes |
| < 5.0 | No |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Primary channel — GitHub Private Vulnerability Reporting

Use GitHub's built-in private vulnerability reporting:

https://github.com/bernardladenthin/java-llama.cpp/security/advisories/new

This channel is private and visible only to maintainers. It is the preferred method.

### Secondary channel — maintainer email

If you cannot use the GitHub advisory form, you may contact the maintainer by email. The address associated with recent commits is listed in the git log (`git log --format='%ae' -1`). Note that this address is **unconfirmed** as a monitored security contact — GitHub Private Vulnerability Reporting above is preferred.

## Response SLA

We aim to acknowledge vulnerability reports within 14 days of receipt and to provide a remediation timeline within 30 days.

## Disclosure Policy

We follow **coordinated disclosure**:

1. Reporter submits the vulnerability privately.
2. Maintainers confirm and assess severity.
3. A fix is developed and a release date is agreed with the reporter.
4. The fix is released and a GitHub Security Advisory is published simultaneously.
5. The reporter may disclose publicly after the fix is released (or after an agreed embargo period, typically 90 days from report, whichever comes first).

We ask reporters to keep vulnerability details **under embargo** until a fix has been released.
