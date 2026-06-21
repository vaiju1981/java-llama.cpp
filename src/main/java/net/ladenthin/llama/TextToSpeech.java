// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.loader.LlamaLoader;

/**
 * Text-to-speech synthesis over llama.cpp's OuteTTS pipeline. Loads two models — a text-to-codes
 * (OuteTTS) model and a codes-to-speech (WavTokenizer) vocoder — and turns text into a 24&nbsp;kHz
 * mono 16-bit WAV byte stream.
 *
 * <p>This is a separate native type from {@link LlamaModel} because TTS is a two-model pipeline that
 * does not use the chat/completion server path. Native memory is not GC-managed: use
 * try-with-resources or call {@link #close()} explicitly.
 *
 * <pre>{@code
 * try (TextToSpeech tts = new TextToSpeech(
 *         "models/OuteTTS-0.2-500M.gguf", "models/WavTokenizer.gguf")) {
 *     byte[] wav = tts.synthesize("Hello from llama dot c p p.");
 *     Files.write(Paths.get("out.wav"), wav);
 * }
 * }</pre>
 *
 * <p><b>Known limitation:</b> numeric digits in the input are dropped (number-to-words romanization
 * is not yet ported); spell numbers out for now. Uses the built-in default speaker profile.
 */
public final class TextToSpeech implements AutoCloseable {

    static {
        LlamaLoader.initialize();
    }

    private long handle;

    /**
     * Load the TTS pipeline CPU-only.
     *
     * @param ttcModelPath path to the text-to-codes (OuteTTS) GGUF
     * @param vocoderModelPath path to the codes-to-speech (WavTokenizer) vocoder GGUF
     */
    public TextToSpeech(String ttcModelPath, String vocoderModelPath) {
        this(ttcModelPath, vocoderModelPath, 0, 0);
    }

    /**
     * Load the TTS pipeline.
     *
     * @param ttcModelPath path to the text-to-codes (OuteTTS) GGUF
     * @param vocoderModelPath path to the codes-to-speech (WavTokenizer) vocoder GGUF
     * @param gpuLayers number of layers to offload to the GPU (0 = CPU only)
     * @param threads CPU threads for the spectral DSP (0 = a small default)
     */
    public TextToSpeech(String ttcModelPath, String vocoderModelPath, int gpuLayers, int threads) {
        this.handle = loadNative(ttcModelPath, vocoderModelPath, gpuLayers, threads);
    }

    /**
     * Synthesize speech with default sampling (top-k 4, seed 0, up to 4096 code tokens).
     *
     * @param text the text to speak
     * @return a 24&nbsp;kHz mono 16-bit WAV byte stream
     */
    public byte[] synthesize(String text) {
        return synthesize(text, 4096, 4, 0);
    }

    /**
     * Synthesize speech with explicit sampling parameters.
     *
     * @param text the text to speak
     * @param maxCodeTokens cap on generated audio-code tokens (longer = longer audio)
     * @param topK top-k sampling cutoff for the code model
     * @param seed sampler seed
     * @return a 24&nbsp;kHz mono 16-bit WAV byte stream
     */
    public byte[] synthesize(String text, int maxCodeTokens, int topK, int seed) {
        if (handle == 0L) {
            throw new IllegalStateException("TextToSpeech is closed");
        }
        return synthesizeNative(handle, text, maxCodeTokens, topK, seed);
    }

    @Override
    public synchronized void close() {
        if (handle != 0L) {
            deleteNative(handle);
            handle = 0L;
        }
    }

    private static native long loadNative(String ttcModelPath, String vocoderModelPath, int gpuLayers, int threads);

    private static native byte[] synthesizeNative(long handle, String text, int maxCodeTokens, int topK, int seed);

    private static native void deleteNative(long handle);
}
