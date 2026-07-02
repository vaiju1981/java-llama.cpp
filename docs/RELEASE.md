# Release Process

This document is the maintainer-facing release procedure. End users should consult [CHANGELOG.md](../CHANGELOG.md).

> **This repo is a Maven reactor.** The root `pom.xml` is the parent (`net.ladenthin:llama-parent`);
> the two modules (`llama/`, `llama-langchain4j/`) inherit its version **but hardcode it in their
> `<parent><version>`** — there is no `${revision}` single-sourcing, so a version change must touch
> **all three poms in lockstep** or the reactor build fails with
> `Could not find artifact net.ladenthin:llama-parent:pom:{VERSION}`. `mvn versions:set` does all
> three in one command. See the "Version bump" note in [CLAUDE.md](../CLAUDE.md) for the rationale.

> Paste this prompt into a new Claude Code session, fill in the four placeholders, and send it to perform a release.

```
Release `{PROJECT}` to Maven Central.

**Step 1 — Prepare the release (do immediately):**
1. Read the current version from the root `pom.xml` on `main` — it will be `{VERSION}-SNAPSHOT`.
2. Set the release version across **all three reactor poms** in lockstep (root `<version>` +
   `llama/pom.xml` and `llama-langchain4j/pom.xml` `<parent><version>`):
   `mvn -q versions:set -DnewVersion={VERSION} -DgenerateBackupPoms=false`
   Changing only the root pom leaves the children pointing at a non-existent parent and fails the build.
3. In `README.md`, set **every release** dependency example (core Quick Start, the classifier
   table, and the LangChain4j section) to `{VERSION}`, and verify the single **snapshot** example
   (under "Snapshot builds") stays `{VERSION}-SNAPSHOT`.
4. In `llama-langchain4j/README.md`, set the `## Dependency` snippet to `{VERSION}` — it is a
   release snippet and is NOT covered by the root-README edits (it drifts silently otherwise).
5. Finalize `CHANGELOG.md`: rename the `[Unreleased]` heading to `[{VERSION}] - {DATE}`, add a fresh
   empty `[Unreleased]` above it, and update the compare-link footer — add
   `[{VERSION}]: .../compare/v{PREV}...v{VERSION}` and repoint
   `[Unreleased]: .../compare/v{VERSION}...HEAD`.
6. Commit the changes on a branch and open a PR; merge/fast-forward it into `main`.

**Step 2 — Wait for manual confirmation:**
I will (a) create the `v{VERSION}` tag + GitHub release on the merged commit and (b) run the
**Publish** workflow via `workflow_dispatch` with `publish_to_central=true` (the `publish-release`
job is gated on that input **and** the `v*` tag; one reactor `mvn -P release deploy` signs and
publishes the parent pom, `llama`, and `llama-langchain4j` together). Wait for me to confirm the
release is live on Maven Central before proceeding.

**Step 3 — Post-release snapshot bump (after my confirmation):**
On a fresh branch off the updated `main`:
- Bump all three poms to `{NEXT_VERSION}-SNAPSHOT`:
  `mvn -q versions:set -DnewVersion={NEXT_VERSION}-SNAPSHOT -DgenerateBackupPoms=false`
- `README.md` snapshot dependency example → `{NEXT_VERSION}-SNAPSHOT`
  (the release examples — including `llama-langchain4j/README.md` — stay at `{VERSION}`).
Commit and open a PR.

**Placeholders:**

| Placeholder      | Value                                        |
|------------------|----------------------------------------------|
| `{PROJECT}`      | *(project name)*                             |
| `{VERSION}`      | *(release version, e.g. `1.3.0`)*           |
| `{PREV}`         | *(previous release, e.g. `1.2.0`)*          |
| `{NEXT_VERSION}` | *(next snapshot base, e.g. `1.3.1`)*        |
```
