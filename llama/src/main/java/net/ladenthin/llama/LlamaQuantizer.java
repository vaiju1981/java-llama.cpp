// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Objects;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.loader.LlamaLoader;

/**
 * In-JVM GGUF model quantization over llama.cpp's {@code llama_model_quantize} — the Java
 * counterpart of the {@code llama-quantize} CLI tool. Converts a GGUF file to another
 * {@linkplain QuantizationType quantization scheme} without leaving the JVM or shelling out.
 *
 * <pre>{@code
 * LlamaQuantizer.quantize("model-f16.gguf", "model-q4_k_m.gguf", QuantizationType.Q4_K_M);
 * }</pre>
 *
 * <p>Quantizing an <em>already-quantized</em> input (re-quantization) is refused by llama.cpp by
 * default because it degrades quality; opt in explicitly via
 * {@link #quantize(String, String, QuantizationType, int, boolean)} with
 * {@code allowRequantize = true}.</p>
 *
 * <p>Quantization is CPU-bound and can take minutes for large models; it initializes the shared
 * llama backend (a no-op if a {@link LlamaModel} is already loaded) and may safely run while
 * models are loaded in the same JVM.</p>
 */
public final class LlamaQuantizer {

    static {
        LlamaLoader.initialize();
    }

    private LlamaQuantizer() {}

    /**
     * Quantize {@code inputPath} to {@code outputPath} with default settings (all available
     * hardware threads, no re-quantization of already-quantized inputs).
     *
     * @param inputPath  the source GGUF (typically F32/F16/BF16)
     * @param outputPath the destination GGUF to write
     * @param type       the target quantization scheme
     * @throws net.ladenthin.llama.exception.LlamaException if quantization fails (missing input,
     *         unwritable output, re-quantization without opt-in, unsupported tensor layout, …)
     */
    public static void quantize(String inputPath, String outputPath, QuantizationType type) {
        quantize(inputPath, outputPath, type, 0, false);
    }

    /**
     * Quantize {@code inputPath} to {@code outputPath} with explicit thread count and
     * re-quantization opt-in.
     *
     * @param inputPath       the source GGUF
     * @param outputPath      the destination GGUF to write
     * @param type            the target quantization scheme
     * @param threads         quantization threads; {@code <= 0} uses all hardware threads
     * @param allowRequantize permit quantizing tensors that are not F32/F16 (re-quantizing an
     *                        already-quantized model — lossy on top of lossy, quality degrades)
     * @throws net.ladenthin.llama.exception.LlamaException if quantization fails
     */
    public static void quantize(
            String inputPath, String outputPath, QuantizationType type, int threads, boolean allowRequantize) {
        Objects.requireNonNull(inputPath, "inputPath");
        Objects.requireNonNull(outputPath, "outputPath");
        Objects.requireNonNull(type, "type");
        quantizeNative(inputPath, outputPath, type.getFtypeValue(), threads, allowRequantize);
    }

    private static native void quantizeNative(
            String inputPath, String outputPath, int ftype, int threads, boolean allowRequantize);
}
