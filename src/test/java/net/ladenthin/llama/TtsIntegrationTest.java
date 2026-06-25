// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Real-model coverage for {@link TextToSpeech} (OuteTTS audio output, llama.cpp {@code llama-tts}
 * pipeline). Loads the two-model TTS pipeline and synthesizes a short clip, checking the WAV
 * container is well-formed.
 *
 * <p>Self-skips when {@link TestConstants#PROP_TTS_TTC_MODEL} or
 * {@link TestConstants#PROP_TTS_VOCODER_MODEL} is unset or its file is missing, so it runs only where
 * the (large) OuteTTS + WavTokenizer GGUFs have been staged.
 */
public class TtsIntegrationTest {

    @Test
    @DisplayName("synthesize() returns a well-formed 16-bit WAV byte stream")
    @Timeout(value = 300_000, unit = TimeUnit.MILLISECONDS)
    public void synthesizesWellFormedWav() {
        String ttc = System.getProperty(TestConstants.PROP_TTS_TTC_MODEL);
        String vocoder = System.getProperty(TestConstants.PROP_TTS_VOCODER_MODEL);
        Assumptions.assumeTrue(
                ttc != null && !ttc.isEmpty(), "TTS model not set (-D" + TestConstants.PROP_TTS_TTC_MODEL + "=...)");
        Assumptions.assumeTrue(
                vocoder != null && !vocoder.isEmpty(),
                "TTS vocoder not set (-D" + TestConstants.PROP_TTS_VOCODER_MODEL + "=...)");
        Assumptions.assumeTrue(new File(ttc).exists(), "TTS model file missing: " + ttc);
        Assumptions.assumeTrue(new File(vocoder).exists(), "TTS vocoder file missing: " + vocoder);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, 0);
        try (TextToSpeech tts = new TextToSpeech(ttc, vocoder, gpuLayers, 0)) {
            byte[] wav = tts.synthesize("hello from llama");

            assertNotNull(wav, "WAV bytes must not be null");
            assertTrue(wav.length > 44, "WAV must carry a header plus samples; got " + wav.length + " bytes");
            assertEquals('R', (char) wav[0]);
            assertEquals('I', (char) wav[1]);
            assertEquals('F', (char) wav[2]);
            assertEquals('F', (char) wav[3]);
            assertEquals('W', (char) wav[8]);
            assertEquals('A', (char) wav[9]);
            assertEquals('V', (char) wav[10]);
            assertEquals('E', (char) wav[11]);
        }
    }
}
