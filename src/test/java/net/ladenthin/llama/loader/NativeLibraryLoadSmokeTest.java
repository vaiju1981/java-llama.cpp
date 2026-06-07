// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

/**
 * Model-free smoke test that the bundled native library actually loads and its
 * {@code JNI_OnLoad} resolves every Java class it looks up by name.
 *
 * <p>Forcing {@code LlamaModel.<clinit>} runs
 * {@code LlamaLoader.initialize() -> System.load() -> JNI_OnLoad}, which calls
 * {@code FindClass(...)} for the JNI-referenced classes ({@code LlamaException},
 * {@code LogLevel}, {@code LogFormat}, ...). No GGUF model is required, so this
 * catches the two failure modes that the model-gated tests cannot exercise when
 * models are absent (e.g. in a restricted-network sandbox):
 *
 * <ul>
 *   <li>a wrong native-resource path in {@link LlamaLoader} (lib not found), and</li>
 *   <li>a stale {@code FindClass} FQN in {@code jllama.cpp} after a Java package
 *       move (lib loads but {@code JNI_OnLoad} throws
 *       {@code NoClassDefFoundError}).</li>
 * </ul>
 *
 * <p>Both bugs shipped once on this branch precisely because they only surface
 * when the library is loaded — see the regression history in {@code CLAUDE.md}.
 *
 * <p>The test self-skips when {@code libjllama} is not on the classpath (a
 * pure-Java checkout with no native build), so a plain {@code mvn test} stays
 * green without a CMake build; CI's {@code test-java-*} jobs and any local build
 * have the library and run it for real. The presence check uses the canonical
 * resource layout directly (not {@link LlamaLoader#getNativeResourcePath()}) so
 * a regression in that method cannot silently skip this guard.
 */
@ClaudeGenerated(
        purpose = "Model-free native-load smoke: force LlamaModel.<clinit> so System.load + JNI_OnLoad "
                + "run and resolve every FindClass'd Java class. Guards against native-resource-path and "
                + "stale-JNI-FQN regressions that only appear when the library is actually loaded; skips "
                + "cleanly when libjllama is not on the classpath.")
class NativeLibraryLoadSmokeTest {

    private static boolean nativeLibraryOnClasspath() {
        String resource = "/net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/"
                + System.mapLibraryName("jllama");
        return NativeLibraryLoadSmokeTest.class.getResource(resource) != null;
    }

    @Test
    void loadingNativeLibraryRunsJniOnLoadWithoutError() {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping native-load smoke");
        assertDoesNotThrow(
                () -> Class.forName("net.ladenthin.llama.LlamaModel"),
                "LlamaModel.<clinit> must load the native library and JNI_OnLoad must resolve "
                        + "every FindClass'd Java class");
    }
}
