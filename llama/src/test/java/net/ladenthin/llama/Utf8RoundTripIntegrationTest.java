// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Integration tests pinning that text payloads cross the JNI boundary as <em>standard</em>
 * UTF-8 in both directions.
 *
 * <p>Two failure modes are guarded:
 * <ul>
 *   <li>{@code NewStringUTF} expects Modified UTF-8, where supplementary-plane characters
 *       (every 4-byte emoji sequence) are CESU-8 surrogate pairs — passing standard UTF-8
 *       through it mangles emoji on HotSpot and aborts under Android CheckJNI. The native
 *       layer therefore builds response strings via {@code String(byte[], "UTF-8")}
 *       ({@code utf8_to_jstring_impl}); {@link #applyTemplate_supplementaryPlane_roundTrips()}
 *       exercises that path deterministically (no generation involved).</li>
 *   <li>Streamed chunks must never split a multi-byte codepoint. The upstream server core
 *       holds back incomplete UTF-8 at the end of the generated text
 *       ({@code server_context::process_token}); the streaming tests assert no chunk carries
 *       a replacement character or a lone surrogate, whatever the model generates.</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Pin standard-UTF-8 round-trip correctness across the JNI boundary: "
                + "deterministic applyTemplate echo of emoji/CJK input and well-formed "
                + "(no replacement char, no lone surrogate) streamed chunks.")
public class Utf8RoundTripIntegrationTest {

    /** BMP + supplementary-plane sample: CJK (3-byte UTF-8) and emoji (4-byte UTF-8). */
    private static final String UNICODE_SAMPLE = "你好 😀🚀 grüße";

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model not found, skipping Utf8RoundTripIntegrationTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers)
                .setFit(false));
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /** Asserts {@code text} is well-formed UTF-16: no lone surrogate, no replacement char. */
    private static void assertWellFormed(String text, String context) {
        assertFalse(text.indexOf('�') >= 0, context + " contains a U+FFFD replacement character: " + text);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isHighSurrogate(c)) {
                assertTrue(
                        i + 1 < text.length() && Character.isLowSurrogate(text.charAt(i + 1)),
                        context + " contains a lone high surrogate at index " + i + ": " + text);
                i++;
            } else {
                assertFalse(
                        Character.isLowSurrogate(c),
                        context + " contains a lone low surrogate at index " + i + ": " + text);
            }
        }
    }

    /**
     * The rendered chat template must echo emoji (supplementary plane) and CJK input
     * byte-correct through the native jstring construction — deterministic, no sampling.
     */
    @Test
    public void applyTemplate_supplementaryPlane_roundTrips() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", UNICODE_SAMPLE));
        InferenceParameters params = new InferenceParameters("").withMessages(null, messages);

        String prompt = model.applyTemplate(params);
        assertTrue(
                prompt.contains(UNICODE_SAMPLE),
                "applyTemplate must round-trip supplementary-plane characters, got: " + prompt);
        assertWellFormed(prompt, "applyTemplate result");
    }

    /**
     * Every streamed chunk must be well-formed regardless of what the model generates: the
     * native side may only flush completed codepoints to the iterator.
     */
    @Test
    public void streaming_chunksAreAlwaysWellFormedUtf8() {
        InferenceParameters params = new InferenceParameters("Repeat this exactly: " + UNICODE_SAMPLE + "\n")
                .withNPredict(48)
                .withSeed(42)
                .withTemperature(0.0f);
        StringBuilder all = new StringBuilder();
        for (LlamaOutput output : model.generate(params)) {
            assertWellFormed(output.text, "streamed chunk");
            all.append(output.text);
        }
        assertWellFormed(all.toString(), "concatenated stream");
    }
}
