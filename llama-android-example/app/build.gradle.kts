// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Injected by CI (-PjllamaVersion=<reactor version>) so the example always runs against
// the AAR + facade that were just built and published to mavenLocal. Falls back to the
// last release when built by hand / in Android Studio without the property.
val jllamaVersion: String =
    providers.gradleProperty("jllamaVersion").getOrElse("5.0.6")

// -- Release signing (Play upload key) -------------------------------------------------
// The APK/AAB signing key is a Java keystore (PKCS12/JKS + RSA), NOT the Maven Central
// GPG key — different cryptosystem, different tooling. For "as close to a real Play
// release as possible" this reads a dedicated UPLOAD keystore (Play App Signing manages
// the final app-signing key) from env vars (CI secrets) or -P properties (local).
// When no keystore is provided the release build falls back to debug signing, so PR CI,
// forks and local builds still produce an installable APK without any secret.
fun secret(env: String, prop: String): String? =
    System.getenv(env) ?: (findProperty(prop) as String?)

val uploadStorePath: String? = secret("JLLAMA_UPLOAD_STORE_FILE", "jllamaUploadStoreFile")
val hasUploadKeystore: Boolean = !uploadStorePath.isNullOrBlank() && file(uploadStorePath).exists()

android {
    namespace = "net.ladenthin.llama.example"
    compileSdk = 35

    defaultConfig {
        applicationId = "net.ladenthin.llama.example"
        minSdk = 28 // AAR floor (bionic weak-symbol gate for posix_spawn); AGP enforces it.
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // arm64 = real phones; x86_64 = the CI emulator (and x86_64 Android hardware).
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        if (hasUploadKeystore) {
            create("upload") {
                storeFile = file(uploadStorePath!!)
                storePassword = secret("JLLAMA_UPLOAD_STORE_PASSWORD", "jllamaUploadStorePassword")
                keyAlias = secret("JLLAMA_UPLOAD_KEY_ALIAS", "jllamaUploadKeyAlias")
                keyPassword = secret("JLLAMA_UPLOAD_KEY_PASSWORD", "jllamaUploadKeyPassword")
            }
        }
    }

    buildTypes {
        release {
            // R8 on; the AAR ships consumer ProGuard rules that keep the JNI surface,
            // so no app-side keep rules are needed (proguard-rules.pro is a stub).
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            signingConfig =
                if (hasUploadKeystore) {
                    signingConfigs.getByName("upload")
                } else {
                    logger.lifecycle("No upload keystore configured -> release signed with the debug key.")
                    signingConfigs.getByName("debug")
                }
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // The binding (classes + libjllama.so for arm64-v8a + x86_64) and the Kotlin
    // coroutines facade (generateChatFlow). The facade's core dependency is
    // provided-scope, so Gradle does NOT pull the fat desktop JAR — the AAR supplies
    // net.ladenthin.llama.* on Android.
    implementation("net.ladenthin:llama-android:$jllamaVersion")
    implementation("net.ladenthin:llama-kotlin:$jllamaVersion")

    // Coroutines (matches the facade's kotlinx-coroutines 1.11.0).
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")

    // Jetpack Compose UI (BOM pins the mutually consistent runtime versions).
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // On-emulator instrumentation (build-android-example CI job).
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:core:1.6.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
