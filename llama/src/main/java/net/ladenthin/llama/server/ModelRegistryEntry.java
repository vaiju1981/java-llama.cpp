// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.jspecify.annotations.Nullable;

/**
 * One entry in the {@link ModelRegistry}: a stable model name and where it came from / lives on disk.
 *
 * <p>Immutable. Built via {@link Builder} or parsed from the registry manifest JSON. Round-trips
 * through {@link #toJsonNode()} so future manifest fields can be added without breaking older code.
 */
public final class ModelRegistryEntry {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final String name;
    private final @Nullable String localPath;
    private final @Nullable String sourceUrl;
    private final @Nullable String quantization;
    private final long sizeBytes;
    private final List<String> aliases;
    private final long pulledAt;

    private ModelRegistryEntry(Builder b) {
        this.name = b.name;
        this.localPath = b.localPath;
        this.sourceUrl = b.sourceUrl;
        this.quantization = b.quantization;
        this.sizeBytes = b.sizeBytes;
        this.aliases =
                Collections.unmodifiableList(new ArrayList<>(b.aliases));
        this.pulledAt = b.pulledAt;
    }

    /**
     * Returns the stable model name.
     *
     * @return the stable model name (also the registry key)
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the local GGUF path.
     *
     * @return local GGUF path, or {@code null} when only a source URL is known
     */
    public @Nullable String getLocalPath() {
        return localPath;
    }

    /**
     * Returns the source URL the model was pulled from.
     *
     * @return the URL the model was pulled from, or {@code null} for a local-only entry
     */
    public @Nullable String getSourceUrl() {
        return sourceUrl;
    }

    /**
     * Returns the quantization label.
     *
     * @return quantization label (e.g. {@code Q4_K_M}), or {@code null} when unknown
     */
    public @Nullable String getQuantization() {
        return quantization;
    }

    /**
     * Returns the downloaded file size in bytes.
     *
     * @return downloaded file size in bytes (0 when unknown)
     */
    public long getSizeBytes() {
        return sizeBytes;
    }

    /**
     * Returns the alternative names that also resolve to this entry.
     *
     * @return alternative names that also resolve to this entry
     */
    public List<String> getAliases() {
        return aliases;
    }

    /**
     * Returns the epoch millis when the entry was registered.
     *
     * @return epoch millis when the entry was registered, or 0 when unknown
     */
    public long getPulledAt() {
        return pulledAt;
    }

    /**
     * Returns this entry as a registry-manifest JSON object.
     *
     * @return this entry as a registry-manifest JSON object
     */
    public JsonNode toJsonNode() {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("name", name);
        if (localPath != null) {
            node.put("local_path", localPath);
        }
        if (sourceUrl != null) {
            node.put("source_url", sourceUrl);
        }
        if (quantization != null) {
            node.put("quantization", quantization);
        }
        node.put("size_bytes", sizeBytes);
        ArrayNode aliasArray = node.putArray("aliases");
        for (String alias : aliases) {
            aliasArray.add(alias);
        }
        node.put("pulled_at", pulledAt);
        return node;
    }

    /**
     * Parse a registry-manifest entry object. Missing optional fields default to {@code null}/empty.
     *
     * @param node the entry JSON object
     * @return the parsed entry
     */
    public static ModelRegistryEntry fromJson(JsonNode node) {
        Builder b = new Builder(node.path("name").asText());
        b.localPath(textOrNull(node, "local_path"));
        b.sourceUrl(textOrNull(node, "source_url"));
        b.quantization(textOrNull(node, "quantization"));
        b.sizeBytes(node.path("size_bytes").asLong(0L));
        if (node.has("aliases") && node.path("aliases").isArray()) {
            List<String> aliases = new ArrayList<>();
            for (JsonNode alias : node.path("aliases")) {
                aliases.add(alias.asText());
            }
            b.aliases(aliases);
        }
        b.pulledAt(node.path("pulled_at").asLong(0L));
        return b.build();
    }

    private static @Nullable String textOrNull(JsonNode node, String field) {
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.asText();
    }

    /** Builder for {@link ModelRegistryEntry}. */
    public static final class Builder {
        private final String name;
        private @Nullable String localPath;
        private @Nullable String sourceUrl;
        private @Nullable String quantization;
        private long sizeBytes;
        private List<String> aliases = Collections.emptyList();
        private long pulledAt;

        /** @param name the stable model name (required) */
        public Builder(String name) {
            this.name = name;
        }

        public Builder localPath(@Nullable String localPath) {
            this.localPath = localPath;
            return this;
        }

        public Builder sourceUrl(@Nullable String sourceUrl) {
            this.sourceUrl = sourceUrl;
            return this;
        }

        public Builder quantization(@Nullable String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder sizeBytes(long sizeBytes) {
            this.sizeBytes = sizeBytes;
            return this;
        }

        public Builder aliases(@Nullable List<String> aliases) {
            this.aliases = aliases == null ? Collections.emptyList() : aliases;
            return this;
        }

        public Builder pulledAt(long pulledAt) {
            this.pulledAt = pulledAt;
            return this;
        }

        public ModelRegistryEntry build() {
            return new ModelRegistryEntry(this);
        }
    }
}
