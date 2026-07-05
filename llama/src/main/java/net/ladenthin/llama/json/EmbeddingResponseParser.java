// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Pure JSON transforms for the OpenAI-compatible embeddings wire format used by the typed
 * batch-embedding API ({@code net.ladenthin.llama.LlamaModel#embed(java.util.Collection)}).
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code EmbeddingResponseParserTest}).
 *
 * <p>The native OAI embeddings response has the shape:
 * <pre>{@code
 * {
 *   "object": "list",
 *   "data": [
 *     {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0},
 *     {"object": "embedding", "embedding": [0.3, 0.4, ...], "index": 1}
 *   ]
 * }
 * }</pre>
 */
public class EmbeddingResponseParser {

    /** Creates a new {@link EmbeddingResponseParser}. */
    public EmbeddingResponseParser() {}

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Parse the embedding vectors from a raw OAI-format JSON response string. Delegates to
     * {@link #parse(JsonNode)} after a single {@code readTree} call.
     *
     * @param json raw OAI embeddings response JSON
     * @return one float vector per input, ordered by the response's {@code index} field;
     *         empty list on parse failure
     */
    public List<float[]> parse(String json) {
        try {
            return parse(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    /**
     * Parse the embedding vectors from a pre-parsed OAI-format response node. Entries are
     * ordered by their {@code "index"} field (falling back to array position when absent), so
     * the result lines up with the request's input order. Entries whose {@code "embedding"}
     * is not a numeric array are skipped.
     *
     * @param node pre-parsed OAI embeddings response
     * @return one float vector per parsed entry, ordered by index; empty list when
     *         {@code "data"} is absent or not an array
     */
    public List<float[]> parse(JsonNode node) {
        List<IndexedVector> vectors = new ArrayList<IndexedVector>();
        JsonNode data = node.path("data");
        if (!data.isArray()) {
            return new ArrayList<float[]>();
        }
        int position = 0;
        for (JsonNode entry : data) {
            JsonNode embedding = entry.path("embedding");
            if (!embedding.isArray()) {
                position++;
                continue;
            }
            float[] vector = new float[embedding.size()];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = (float) embedding.get(i).asDouble(0.0);
            }
            vectors.add(new IndexedVector(entry.path("index").asInt(position), vector));
            position++;
        }
        // Stable sort by the response's index field so the vectors line up with the
        // request's input order even if the native side reordered completions.
        vectors.sort((a, b) -> Integer.compare(a.index, b.index));
        List<float[]> ordered = new ArrayList<float[]>(vectors.size());
        for (IndexedVector entry : vectors) {
            ordered.add(entry.vector);
        }
        return ordered;
    }

    /** One parsed embedding paired with the response's {@code index} field for reordering. */
    private static final class IndexedVector {
        final int index;
        final float[] vector;

        IndexedVector(int index, float[] vector) {
            this.index = index;
            this.vector = vector;
        }
    }

    /**
     * Build the OAI batch-embedding request body {@code {"input": [prompt, ...]}} for
     * the native embeddings endpoint.
     *
     * @param prompts the strings to embed, in order; must not be empty
     * @return the request JSON string
     */
    public String toBatchRequestJson(Collection<String> prompts) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        ArrayNode input = root.putArray("input");
        for (String prompt : prompts) {
            input.add(prompt);
        }
        return root.toString();
    }
}
