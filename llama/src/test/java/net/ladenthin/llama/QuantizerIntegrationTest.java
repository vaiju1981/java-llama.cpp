// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration tests for {@link LlamaQuantizer} against a real GGUF. Uses the small draft model
 * (AMD-Llama-135m, Q2_K) so the re-quantization round trip stays fast; the produced Q4_0 file is
 * then loaded and asked for a short completion, proving the output is a valid, loadable model.
 */
@ClaudeGenerated(
        purpose = "Exercise llama_model_quantize end to end over the JNI surface: successful "
                + "re-quantization producing a loadable GGUF, the default refusal to requantize "
                + "an already-quantized input, and the missing-input error path.")
public class QuantizerIntegrationTest {

    @TempDir
    static Path tempDir;

    private static void assumeDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping QuantizerIntegrationTest");
    }

    @Test
    public void quantize_producesLoadableModel() throws Exception {
        assumeDraftModel();
        Path output = tempDir.resolve("draft-q4_0.gguf");

        LlamaQuantizer.quantize(TestConstants.DRAFT_MODEL_PATH, output.toString(), QuantizationType.Q4_0, 0, true);

        assertTrue(Files.exists(output), "quantized output must exist");
        assertTrue(Files.size(output) > 10_000_000L, "quantized 135M model should be well above 10 MB");

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setModel(output.toString())
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {
            String completion = model.complete(
                    new InferenceParameters("def main():").withNPredict(4).withTemperature(0.0f));
            assertNotNull(completion, "quantized model must be able to complete");
        }
    }

    /** Re-quantizing an already-quantized GGUF without the explicit opt-in must fail loudly. */
    @Test
    public void quantize_requantizeWithoutOptIn_throws() {
        assumeDraftModel();
        Path output = tempDir.resolve("draft-requant-refused.gguf");
        assertThrows(
                LlamaException.class,
                () -> LlamaQuantizer.quantize(
                        TestConstants.DRAFT_MODEL_PATH, output.toString(), QuantizationType.Q4_0));
    }

    @Test
    public void quantize_missingInput_throws() {
        assumeDraftModel(); // gates on the native lib being present in this environment
        Path output = tempDir.resolve("never-written.gguf");
        assertThrows(
                LlamaException.class,
                () -> LlamaQuantizer.quantize(
                        "models/does-not-exist.gguf", output.toString(), QuantizationType.Q8_0, 0, true));
    }
}
