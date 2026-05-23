// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

class TestConstants {

	/** System property to override GPU layers used in tests. */
	static final String PROP_TEST_NGL = LlamaSystemProperties.PREFIX + ".test.ngl";

	static final int DEFAULT_TEST_NGL = 43;

	/** Path to the main text generation model used in tests. */
	static final String MODEL_PATH = "models/codellama-7b.Q2_K.gguf";

	/** Path to the draft model used for speculative decoding tests. */
	static final String DRAFT_MODEL_PATH = "models/AMD-Llama-135m-code.Q2_K.gguf";

	/** Path to the Qwen3 thinking model used for reasoning budget tests. */
	static final String REASONING_MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf";

	/**
	 * System property holding a path to a Nomic embedding model
	 * ({@code nomic-embed-text-v1.5.f16.gguf} or a compatible BERT-family encoder).
	 * Used by {@link LlamaEmbeddingsTest#testNomicEmbedLoads} to confirm upstream
	 * issue #98 (BERT-encoder result_output assertion) stays resolved.
	 * When the property is unset the test self-skips.
	 */
	static final String PROP_NOMIC_MODEL_PATH = LlamaSystemProperties.PREFIX + ".nomic.path";

	/** Expected embedding dimension of nomic-embed-text-v1.5 (hidden size = 768). */
	static final int NOMIC_EMBED_DIM = 768;

	/**
	 * System property holding a path to a vision-capable model GGUF. Consumed by
	 * {@code MultimodalIntegrationTest} (closes #103 / #34). The CI default is the
	 * SmolVLM-500M Q8_0 GGUF; the test self-skips when the property is unset or
	 * the file is missing.
	 */
	static final String PROP_VISION_MODEL_PATH = LlamaSystemProperties.PREFIX + ".vision.model";

	/** System property holding a path to the matching mmproj GGUF for the vision model. */
	static final String PROP_VISION_MMPROJ_PATH = LlamaSystemProperties.PREFIX + ".vision.mmproj";

	/**
	 * System property holding a path to a CC0 / public-domain image used as the
	 * visual prompt in {@code MultimodalIntegrationTest}. A small Wikimedia
	 * Commons image is the CI default; any png/jpeg/webp/gif works locally.
	 */
	static final String PROP_VISION_IMAGE_PATH = LlamaSystemProperties.PREFIX + ".vision.image";

}
