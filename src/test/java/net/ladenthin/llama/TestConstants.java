// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.loader.LlamaSystemProperties;

public class TestConstants {

    /** System property to override GPU layers used in tests. */
    public static final String PROP_TEST_NGL = LlamaSystemProperties.PREFIX + ".test.ngl";

    public static final int DEFAULT_TEST_NGL = 43;

    /** Path to the main text generation model used in tests. */
    public static final String MODEL_PATH = "models/codellama-7b.Q2_K.gguf";

    /** Path to the draft model used for speculative decoding tests. */
    public static final String DRAFT_MODEL_PATH = "models/AMD-Llama-135m-code.Q2_K.gguf";

    /** Path to the Qwen3 thinking model used for reasoning budget tests. */
    public static final String REASONING_MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf";

    /** Path to the reranking model used in tests (loaded with {@code enableReranking()}). */
    public static final String RERANKING_MODEL_PATH = "models/jina-reranker-v1-tiny-en-Q4_0.gguf";

    /** System property overriding the GGUF used by the real tool-calling integration tests. */
    public static final String PROP_TOOL_MODEL_PATH = LlamaSystemProperties.PREFIX + ".tool.model";

    /** Qwen2.5 tool-capable model used by upstream llama.cpp's blocking and streaming tests. */
    public static final String DEFAULT_TOOL_MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";

    /**
     * System property holding a path to a Nomic embedding model
     * ({@code nomic-embed-text-v1.5.f16.gguf} or a compatible BERT-family encoder).
     * Used by {@link LlamaEmbeddingsTest#testNomicEmbedLoads} to confirm upstream
     * issue #98 (BERT-encoder result_output assertion) stays resolved.
     * When the property is unset the test self-skips.
     */
    public static final String PROP_NOMIC_MODEL_PATH = LlamaSystemProperties.PREFIX + ".nomic.path";

    /** Expected embedding dimension of nomic-embed-text-v1.5 (hidden size = 768). */
    public static final int NOMIC_EMBED_DIM = 768;

    /**
     * System property holding a path to a vision-capable model GGUF. Consumed by
     * {@code MultimodalIntegrationTest} (closes #103 / #34). The CI default is the
     * SmolVLM-500M Q8_0 GGUF; the test self-skips when the property is unset or
     * the file is missing.
     */
    public static final String PROP_VISION_MODEL_PATH = LlamaSystemProperties.PREFIX + ".vision.model";

    /** System property holding a path to the matching mmproj GGUF for the vision model. */
    public static final String PROP_VISION_MMPROJ_PATH = LlamaSystemProperties.PREFIX + ".vision.mmproj";

    /**
     * System property holding a path to an image used as the visual prompt in
     * {@code MultimodalIntegrationTest}. When unset the test falls back to
     * {@link #DEFAULT_VISION_IMAGE_PATH}, which points at a small image
     * committed under {@code src/test/resources/images/}. Any png/jpeg/webp/gif
     * works; the matching extension drives MIME detection in
     * {@code ContentPart.imageFile(Path)}.
     */
    public static final String PROP_VISION_IMAGE_PATH = LlamaSystemProperties.PREFIX + ".vision.image";

    /**
     * Path used by {@code MultimodalIntegrationTest} when
     * {@link #PROP_VISION_IMAGE_PATH} is unset. Points at the committed test
     * resource so the test needs no network access for the visual prompt.
     */
    public static final String DEFAULT_VISION_IMAGE_PATH = "src/test/resources/images/test-image.jpg";
}
