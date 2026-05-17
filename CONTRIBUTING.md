# Contributing to java-llama.cpp

Thank you for your interest in contributing! This document explains how to build the project, file issues, submit pull requests, and what we expect from contributors.

## How to build and run

Prerequisites: Java 11+, Maven 3.x, CMake 3.22+, and a C++17-capable compiler.

```bash
mvn compile                          # generate JNI headers
cmake -B build && cmake --build build --config Release
mvn test
mvn package
```

See [README.md](README.md) for installation, Android, GPU (CUDA/Metal/Vulkan), and runtime usage. See [CLAUDE.md](CLAUDE.md) for the full maintainer guide (upstream-llama.cpp upgrade procedure, native helper architecture, per-version breaking-change table).

## Reporting issues

- Bugs, feature requests, and questions: <https://github.com/bernardladenthin/java-llama.cpp/issues>
- **Security vulnerabilities must not be filed on the public issue tracker.** See [SECURITY.md](SECURITY.md) for the private reporting channel.

When reporting a bug, include: OS and architecture, `java -version` output, the llama.cpp build tag the library was compiled against, a minimal reproduction (model, parameters, code), and the full stack trace.

## Pull request workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, including tests (see [Test policy](#test-policy)).
3. Open a pull request against `bernardladenthin/java-llama.cpp:main`.
4. Ensure CI is green (build, tests, CodeQL, Claude Code review).
5. Respond to review comments by pushing follow-up commits to the same branch — **do not force-push** while a PR is under review; it discards inline-comment context. Force-push is acceptable only to rebase a stale branch onto `main` immediately before merge, after review is approved.
6. A maintainer merges once approved and CI is green.

## Coding standards

Follow the conventions documented in [CLAUDE.md](CLAUDE.md): native helper file split (`json_helpers.hpp` vs `jni_helpers.hpp`), include-order rule for upstream server headers, Javadoc HTML-entity table, and the upstream-llama.cpp upgrade procedure. Java targets 11+. C++ must be C++17-clean. Format C++ with `clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp` before committing.

## Test policy

Every new feature or behaviour change MUST include automated tests; bug fixes SHOULD include a regression test.

- Java tests: JUnit 4, under `src/test/java/net/ladenthin/llama/` and `src/test/java/examples/`. Require a model file (see CI configs for the HuggingFace download).
- C++ unit tests: GoogleTest, under `src/test/cpp/` (`test_json_helpers.cpp`, `test_jni_helpers.cpp`, `test_server.cpp`, `test_utils.cpp`). No JVM or model file required.
- Coverage is reported via JaCoCo and pushed to Coveralls and Codecov on every CI run; new code must not regress coverage on critical paths.

## Commit message convention

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:

- `feat:` — new user-facing feature
- `fix:` — bug fix
- `chore:` — maintenance, dependency bumps, version bumps
- `docs:` — documentation only
- `ci:` — CI / workflow changes
- `refactor:` — code restructuring without behaviour change
- `test:` — test additions or fixes

Scopes are optional (e.g. `fix(publish): quote gate job names`). Keep the subject line under 72 characters.

## Communication channels

- GitHub Issues — <https://github.com/bernardladenthin/java-llama.cpp/issues>
- GitHub Security Advisories — <https://github.com/bernardladenthin/java-llama.cpp/security/advisories/new>

## Code of Conduct

This project adopts the Contributor Covenant. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Releasing

Maintainer-facing release procedure: see [docs/RELEASE.md](docs/RELEASE.md).

## License of contributions

By submitting a pull request you agree that your contribution is made available under the **MIT License** — the same license that governs this repository (see [LICENSE](LICENSE)).
