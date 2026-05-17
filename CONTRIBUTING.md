# Contributing to java-llama.cpp

Thank you for your interest in contributing! This document explains how to build the project, file issues, submit pull requests, and what we expect from contributors.

## Table of Contents

1. [How to Build and Run](#how-to-build-and-run)
2. [Filing Issues](#filing-issues)
3. [Pull Request Workflow](#pull-request-workflow)
4. [Coding Standards](#coding-standards)
5. [Test Policy](#test-policy)
6. [Communication Channels](#communication-channels)
7. [License of Contributions](#license-of-contributions)

---

## How to Build and Run

### Prerequisites

- Java 11 or later
- Maven 3.x
- CMake 3.22 or later
- A C++17-capable compiler (GCC, Clang, or MSVC)

### Java Layer (Maven)

```bash
# Compile Java sources and generate JNI headers (required before CMake build)
mvn compile

# Run all tests (requires a pre-built native library and model files in place)
mvn test

# Run a single test
mvn test -Dtest=LlamaModelTest#testGenerate

# Package a JAR
mvn package
```

### Native Library (CMake)

Run `mvn compile` first to generate the JNI headers, then:

```bash
# CPU-only build
cmake -B build
cmake --build build --config Release

# With CUDA support (Linux)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# With Metal support (macOS)
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release

# With model-download support (libcurl)
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release
```

Built libraries are placed under `src/main/resources/net/ladenthin/llama/{OS}/{ARCH}/`.

### C++ Unit Tests (no JVM or model file required)

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Code Formatting

```bash
# Format C++ source files
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp
```

---

## Filing Issues

Please use the GitHub issue tracker:

- **Bug reports, feature requests, questions:** https://github.com/bernardladenthin/java-llama.cpp/issues

Before opening an issue, search existing issues to avoid duplicates. When reporting a bug, include:

- Operating system and architecture
- Java version (`java -version`)
- llama.cpp build tag the library was compiled against
- A minimal reproduction case (model name, parameters, code snippet)
- Full stack trace or error output

---

## Pull Request Workflow

1. **Fork** the repository on GitHub.
2. Create a **feature branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/my-feature
   ```
3. Make your changes, including tests (see [Test Policy](#test-policy)).
4. Push the branch to your fork and open a **Pull Request** against `bernardladenthin/java-llama.cpp:main`.
5. Describe what the PR changes and why; link any related issue (`Closes #NNN`).
6. Respond to review comments and push follow-up commits to the same branch.
7. A maintainer will merge once the PR is approved and CI is green.

---

## Coding Standards

- Follow the conventions documented in [CLAUDE.md](CLAUDE.md) — it describes the project architecture, include-order rules, helper-file split (`json_helpers.hpp` vs `jni_helpers.hpp`), and Javadoc HTML-entity conventions.
- Java code targets Java 11+.
- C++ code must be compatible with C++17 and compile cleanly with the project's CMake configuration.
- Format C++ files with `clang-format` before committing (see command above).
- Use HTML entities in Javadoc for operators and symbols outside ASCII (see CLAUDE.md for the full table).

---

## Test Policy

> Every new feature or behavior change MUST include automated tests. Pull requests that add or change functionality without corresponding tests will be asked to add tests before merge. Bug fixes SHOULD include a regression test.

- **Java tests** live in `src/test/java/net/ladenthin/llama/` and `src/test/java/examples/`.
- **C++ unit tests** (no JVM required) live in `src/test/cpp/`. Add pure-data transforms to `test_json_helpers.cpp`, JNI bridge helpers to `test_jni_helpers.cpp`, and upstream result types to `test_server.cpp`.
- Tests must pass locally before opening a PR. CI also runs them automatically on push and on pull requests.

---

## Communication Channels

- **GitHub Issues** — bug reports and feature requests: https://github.com/bernardladenthin/java-llama.cpp/issues
- **GitHub Discussions** — general questions and ideas (if enabled on the repository).

---

## License of Contributions

By submitting a pull request you agree that your contribution is made available under the **MIT License** — the same license that governs this repository (see [LICENSE](LICENSE)).
