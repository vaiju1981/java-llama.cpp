// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Throwaway signing project used ONLY by the `verify-signing-key-gradle`
// preflight job in .github/workflows/publish.yml. It mirrors the signing setup
// in llama-android/build.gradle.kts (the AAR publish): the SAME env var names
// and the SAME `useInMemoryPgpKeys(key, passphrase)` call, so running
// `gradle signMakeArtifact` here reproduces the exact Gradle/BouncyCastle
// signing path the AAR uses — the path that failed with a null PGPPrivateKey,
// which the gpg-based `verify-signing-key` job cannot exercise.
//
// It signs a tiny throwaway Zip; the job asserts only that the detached .asc
// was produced. No secret is ever printed (the key/passphrase arrive via env).
plugins {
    base
    signing
}

val signingKey = System.getenv("MAVEN_GPG_PRIVATE_KEY")
val signingPassphrase = System.getenv("MAVEN_GPG_PASSPHRASE")
// Optional: the signing (sub)key id, e.g. 07D2D767. When set, Gradle selects
// that key instead of the primary — needed here because gpg signs with the
// 4096-bit signing subkey while the 2-arg useInMemoryPgpKeys picks the primary
// (whose secret this BouncyCastle can't unlock → the null-PGPPrivateKey NPE).
val signingKeyId = System.getenv("MAVEN_GPG_KEY_ID")
require(!signingKey.isNullOrBlank()) { "MAVEN_GPG_PRIVATE_KEY is empty" }

val makeArtifact = tasks.register<Zip>("makeArtifact") {
    archiveBaseName.set("signing-selftest")
    destinationDirectory.set(layout.buildDirectory)
    from(layout.projectDirectory.file("settings.gradle.kts"))
}

signing {
    if (!signingKeyId.isNullOrBlank()) {
        useInMemoryPgpKeys(signingKeyId, signingKey, signingPassphrase ?: "")
    } else {
        useInMemoryPgpKeys(signingKey, signingPassphrase ?: "")
    }
    sign(makeArtifact.get())
}
