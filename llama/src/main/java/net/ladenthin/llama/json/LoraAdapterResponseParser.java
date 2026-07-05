// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.value.LoraAdapter;

/**
 * Pure JSON transforms for the native LoRA-adapter list and scale-update wire formats.
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code LoraAdapterResponseParserTest}).
 *
 * <p>The native {@code GET /lora-adapters}-equivalent task produces a JSON array:
 * <pre>{@code
 * [
 *   {"id": 0, "path": "adapter.gguf", "scale": 0.5,
 *    "task_name": "classification", "prompt_prefix": ""}
 * ]
 * }</pre>
 *
 * <p>The scale-update request (the upstream {@code POST /lora-adapters} body) is a JSON array
 * of {@code {"id": <int>, "scale": <float>}} objects, built by
 * {@link #toRequestJson(Map)}.
 */
public class LoraAdapterResponseParser {

    /** Creates a new {@link LoraAdapterResponseParser}. */
    public LoraAdapterResponseParser() {}

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Parse the adapter list from a raw JSON array string. Delegates to {@link #parse(JsonNode)}
     * after a single {@code readTree} call.
     *
     * @param json raw JSON array string from the native LoRA-adapter list response
     * @return list of adapters; empty list on parse failure or empty array
     */
    public List<LoraAdapter> parse(String json) {
        try {
            return parse(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    /**
     * Parse the adapter list from a pre-parsed {@link JsonNode} array. Each element carries
     * {@code "id"} (int), {@code "path"} (string), {@code "scale"} (number), {@code "task_name"}
     * (string) and {@code "prompt_prefix"} (string); absent fields fall back to {@code -1},
     * {@code ""}, {@code 0}, {@code ""} and {@code ""} respectively. Returns an empty list when
     * the node is not an array or is empty.
     *
     * @param arr pre-parsed {@link JsonNode} array of adapter objects
     * @return list of adapters; empty list if the node is not an array or is empty
     */
    public List<LoraAdapter> parse(JsonNode arr) {
        List<LoraAdapter> adapters = new ArrayList<LoraAdapter>();
        if (!arr.isArray()) {
            return adapters;
        }
        for (JsonNode entry : arr) {
            adapters.add(new LoraAdapter(
                    entry.path("id").asInt(-1),
                    entry.path("path").asText(""),
                    (float) entry.path("scale").asDouble(0.0),
                    entry.path("task_name").asText(""),
                    entry.path("prompt_prefix").asText("")));
        }
        return adapters;
    }

    /**
     * Build the scale-update request body — a JSON array of {@code {"id", "scale"}} objects in
     * the map's iteration order — matching the upstream {@code POST /lora-adapters} contract:
     * adapters listed in the map get the given scale, all other adapters are set to {@code 0}
     * (disabled) by the native side.
     *
     * @param scales adapter id to new scale; may be empty (disables every adapter)
     * @return the JSON array request string
     * @throws IllegalArgumentException if a scale is NaN or infinite (would not serialize to
     *         valid JSON)
     */
    public String toRequestJson(Map<Integer, Float> scales) {
        com.fasterxml.jackson.databind.node.ArrayNode array = OBJECT_MAPPER.createArrayNode();
        for (Map.Entry<Integer, Float> entry : scales.entrySet()) {
            float scale = entry.getValue();
            if (Float.isNaN(scale) || Float.isInfinite(scale)) {
                throw new IllegalArgumentException(
                        "LoRA scale for adapter " + entry.getKey() + " must be finite, got: " + scale);
            }
            array.addObject().put("id", entry.getKey().intValue()).put("scale", scale);
        }
        return array.toString();
    }
}
