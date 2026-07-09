// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Standalone plain-Gradle example app (NOT a Maven reactor module, like llama-android
// and .github/android-consumer-test). "LLM Service": a KISS, fully-offline local-AI chat
// demo that consumes the shipped net.ladenthin:llama-android AAR + net.ladenthin:llama-kotlin
// coroutines facade. Built + emulator-tested by the build-android-llmservice job in
// .github/workflows/publish.yml; also opens directly in Android Studio.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    // Versions pinned here (KISS: no version catalog). AGP matches the consumer-test
    // fixture; Kotlin matches llama-kotlin (2.4.0).
    plugins {
        id("com.android.application") version "8.7.3"
        id("org.jetbrains.kotlin.android") version "2.4.0"
        id("org.jetbrains.kotlin.plugin.compose") version "2.4.0"
    }
}

dependencyResolutionManagement {
    repositories {
        // In CI the freshly built llama-android AAR + llama-kotlin jar are published to
        // mavenLocal first; for real users these resolve from Maven Central.
        mavenLocal()
        google()
        mavenCentral()
    }
}

rootProject.name = "android-llmservice"
include(":app")
