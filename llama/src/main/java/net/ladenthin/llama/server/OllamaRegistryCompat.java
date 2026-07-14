// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.time.Instant;
import org.jspecify.annotations.Nullable;

/**
 * Adapter that mirrors Ollama's registry/manifest wire shapes so existing Ollama tooling and configs
 * can consume java-llama.cpp's {@link ModelRegistry}. The local {@code models.json} manifest is the
 * source of truth; this class projects it into the shapes Ollama clients expect:
 *
 * <ul>
 *   <li>{@link #listAsOllamaTags(ModelRegistry)} → Ollama's {@code GET /api/tags} body, enumerating
 *       <em>every</em> registered model (Ollama's {@code ollama list}) rather than just the one served
 *       model.</li>
 *   <li>{@link #entryAsOllamaTag(ModelRegistryEntry)} → the single-model {@code /api/tags} entry shape.</li>
 * </ul>
 *
 * <p>Field mapping (jllama → Ollama):
 * <pre>
 *   name            → models[].name / models[].model
 *   pulledAt        → models[].modified_at  (ISO-8601)
 *   sizeBytes       → models[].size
 *   quantization    → models[].details.quantization_level
 *   localPath/url   → models[].digest       (stable, content-derived id; not a real sha256)
 * </pre>
 *
 * <p>The {@code digest} is a stable, non-cryptographic id derived from the model's path or URL. Ollama
 * uses a sha256 of the GGUF blob; jllama does not re-hash the file here, so the digest is only a
 * placeholder for tooling that requires the field to be present. Pure Java, model-free.
 */
public final class OllamaRegistryCompat {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private OllamaRegistryCompat() {}

    /**
     * Project the whole registry into Ollama's {@code GET /api/tags} body.
     *
     * @param registry the local registry
     * @return an Ollama model-list object serialized as JSON
     */
    public static String listAsOllamaTags(ModelRegistry registry) {
        ObjectNode root = MAPPER.createObjectNode();
        ArrayNode models = root.putArray("models");
        for (ModelRegistryEntry entry : registry.list()) {
            models.add(entryAsOllamaTag(entry));
        }
        return root.toString();
    }

    /**
     * Project a single registry entry into Ollama's {@code /api/tags} model shape.
     *
     * @param entry the registry entry
     * @return an Ollama model object serialized as JSON
     */
    public static JsonNode entryAsOllamaTag(ModelRegistryEntry entry) {
        ObjectNode model = MAPPER.createObjectNode();
        model.put("name", entry.getName());
        model.put("model", entry.getName());
        model.put("modified_at", entry.getPulledAt() != 0L ? Instant.ofEpochMilli(entry.getPulledAt()).toString() : "");
        model.put("size", entry.getSizeBytes());
        model.put("digest", digestOf(entry));
        ObjectNode details = model.putObject("details");
        details.put("family", "llama");
        details.put("parameter_size", "");
        String quant = entry.getQuantization();
        details.put("quantization_level", quant != null ? quant : "");
        return model;
    }

    private static String digestOf(ModelRegistryEntry entry) {
        String localPath = entry.getLocalPath();
        String sourceUrl = entry.getSourceUrl();
        String basis = localPath != null ? localPath : (sourceUrl != null ? sourceUrl : entry.getName());
        return "jllama-" + Integer.toHexString(basis.hashCode());
    }
}
