# Release Process

The maintainer-facing release procedure is **centralized in the workspace repo**:
[`../workspace/workflows/release-process.md`](../workspace/workflows/release-process.md).

java-llama.cpp is a **Maven reactor**, so two repo-specific points extend the canonical procedure.

## Reactor version bump (all three poms)

The root `pom.xml` is the parent (`net.ladenthin:llama-parent`); the `llama/` and
`llama-langchain4j/` modules inherit its version **but hardcode it in their `<parent><version>`**
(there is no `${revision}` single-sourcing). A version change — the release strip in Step 1 and the
post-release bump in Step 3 — must touch **all three poms in lockstep**, or the reactor build fails
with `Could not find artifact net.ladenthin:llama-parent:pom:{VERSION}`. Use:

```bash
mvn -q versions:set -DnewVersion={VERSION} -DgenerateBackupPoms=false
```

from the repo root — it updates the root `<version>` plus both children's `<parent><version>` at
once. See the "Version bump" note in [CLAUDE.md](../CLAUDE.md) for the rationale.

## Extra README dependency snippet

Besides the root `README.md`, the `llama-langchain4j/README.md` `## Dependency` section carries a
**release** dependency snippet that must also be set to `{VERSION}` in Step 1 — it is not covered by
the root-README edits and drifts silently otherwise (the release examples stay at `{VERSION}` on the
Step 3 snapshot bump).

One reactor `mvn -P release deploy` signs and publishes the parent pom, `llama`, and
`llama-langchain4j` together at the same version.
