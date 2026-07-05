// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

plugins {
    id("com.android.application")
}

// Injected by CI (-PjllamaVersion=<reactor version>) so the fixture always tests
// the AAR that was just built and published to mavenLocal.
val jllamaVersion: String = providers.gradleProperty("jllamaVersion").get()

android {
    namespace = "net.ladenthin.llama.consumertest"
    compileSdk = 35

    defaultConfig {
        applicationId = "net.ladenthin.llama.consumertest"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // arm64 = real devices; x86_64 = the CI emulator (and x86_64 Android hardware).
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            // Full R8 pass so the AAR's consumer proguard.txt is actually exercised:
            // if the shipped rules were broken, the binding would be stripped or the
            // build would fail here.
            isMinifyEnabled = true
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("net.ladenthin:llama-android:$jllamaVersion")

    // On-emulator instrumentation (test-android-emulator CI job).
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
