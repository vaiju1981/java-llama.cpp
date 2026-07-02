// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import lombok.Builder;
import lombok.Getter;
import net.ladenthin.llama.args.Optimizer;
import org.jspecify.annotations.Nullable;

/**
 * Immutable configuration for a {@link net.ladenthin.llama.LlamaTrainer} fine-tuning run.
 *
 * <p>Build with {@code builder()}; only {@code modelPath} and {@code outputPath} are required, and
 * exactly one of {@code trainingText} / {@code trainingFile} should be set. All other fields default
 * to values that mirror upstream llama.cpp's fine-tuning defaults. The configuration is serialized to
 * JSON via {@link #toJson()} and parsed by the native layer, the same way {@link ModelParameters} and
 * {@link InferenceParameters} cross the JNI boundary.
 */
@Builder
@Getter
public final class TrainingParameters {

    // Base GGUF model to fine-tune.
    private final Path modelPath;

    // Training corpus supplied inline; mutually exclusive with trainingFile.
    private final @Nullable String trainingText;

    // Training corpus read from a file by the native layer; mutually exclusive with trainingText.
    private final @Nullable Path trainingFile;

    // Destination path for the fine-tuned GGUF.
    private final Path outputPath;

    // Number of passes over the corpus (at least 1).
    @Builder.Default
    private final int epochs = 2;

    // Learning rate at the first epoch.
    @Builder.Default
    private final float learningRate = 1e-5f;

    // Minimum learning rate for decay, or -1 to disable decay.
    @Builder.Default
    private final float lrMin = -1f;

    // If > 0, decay the learning rate from learningRate to lrMin over this many epochs.
    @Builder.Default
    private final float decayEpochs = -1f;

    // Weight decay (0 disables it).
    @Builder.Default
    private final float weightDecay = 0f;

    // Optimizer algorithm.
    @Builder.Default
    private final Optimizer optimizer = Optimizer.ADAMW;

    // Context size in tokens, or 0 to use the model's trained context.
    @Builder.Default
    private final int nCtx = 0;

    // Layers to offload to the GPU, or -1 for automatic.
    @Builder.Default
    private final int nGpuLayers = -1;

    // Fraction of the corpus held out for validation.
    @Builder.Default
    private final float valSplit = 0.05f;

    // Logical batch size, or 0 to use the native default.
    @Builder.Default
    private final int nBatch = 0;

    // Physical (micro) batch size, or 0 to use the native default.
    @Builder.Default
    private final int nUbatch = 0;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // Explicit all-args constructor used by the Lombok-generated builder. Declared explicitly (rather
    // than letting @Builder synthesize the package-private one) so Javadoc sees a real constructor and
    // does not emit the "use of default constructor, which does not provide a comment" warning; it is
    // private, so it is not part of the public API and is not doclint-checked.
    private TrainingParameters(
            Path modelPath,
            @Nullable String trainingText,
            @Nullable Path trainingFile,
            Path outputPath,
            int epochs,
            float learningRate,
            float lrMin,
            float decayEpochs,
            float weightDecay,
            Optimizer optimizer,
            int nCtx,
            int nGpuLayers,
            float valSplit,
            int nBatch,
            int nUbatch) {
        this.modelPath = modelPath;
        this.trainingText = trainingText;
        this.trainingFile = trainingFile;
        this.outputPath = outputPath;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.lrMin = lrMin;
        this.decayEpochs = decayEpochs;
        this.weightDecay = weightDecay;
        this.optimizer = optimizer;
        this.nCtx = nCtx;
        this.nGpuLayers = nGpuLayers;
        this.valSplit = valSplit;
        this.nBatch = nBatch;
        this.nUbatch = nUbatch;
    }

    /**
     * Serialize this configuration to the JSON object the native fine-tuning layer expects.
     *
     * @return a compact JSON string
     */
    public String toJson() {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("model_path", modelPath.toString());
        if (trainingText != null) {
            node.put("training_text", trainingText);
        }
        if (trainingFile != null) {
            node.put("training_file", trainingFile.toString());
        }
        node.put("output_path", outputPath.toString());
        node.put("epochs", epochs);
        node.put("learning_rate", learningRate);
        node.put("lr_min", lrMin);
        node.put("decay_epochs", decayEpochs);
        node.put("weight_decay", weightDecay);
        node.put("optimizer", optimizer.getNativeValue());
        node.put("n_ctx", nCtx);
        node.put("n_gpu_layers", nGpuLayers);
        node.put("val_split", valSplit);
        node.put("n_batch", nBatch);
        node.put("n_ubatch", nUbatch);
        return node.toString();
    }
}
