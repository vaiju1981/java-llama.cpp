// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// CI fixture (NOT a shipped project): a minimal AGP app that consumes the
// net.ladenthin:llama-android AAR from mavenLocal and runs a full R8 release
// build — proving Android Studio consumption end-to-end (AAR format, manifest
// minSdk merge, jni packaging, consumer proguard rules) without an emulator.
// Driven by the package-android-aar job in .github/workflows/publish.yml.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        id("com.android.application") version "8.7.3"
    }
}

dependencyResolutionManagement {
    repositories {
        mavenLocal() // the freshly built llama-android AAR is published here first
        google()
        mavenCentral()
    }
}

rootProject.name = "llama-android-consumer-test"
include(":app")
