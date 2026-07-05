// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Standalone Gradle build (NOT a Maven reactor module): Maven cannot produce or
// deploy an artifact with <packaging>aar</packaging> (the only android-maven-plugin
// is long dead), while Gradle's built-in maven-publish can. This build stays
// version-locked to the reactor anyway — build.gradle.kts reads the version from
// the root pom.xml, so `mvn versions:set` remains the single bump point.
rootProject.name = "llama-android"
