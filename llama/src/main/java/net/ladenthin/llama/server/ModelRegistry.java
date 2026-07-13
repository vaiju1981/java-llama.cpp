// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.jspecify.annotations.Nullable;

/**
 * Local model registry: a JSON manifest mapping stable model names to their source and on-disk
 * location. This is the "Ollama-like" naming/lifecycle layer — it lets a model be referenced by a
 * name (e.g. {@code llama3.2}) instead of a raw path/URL, and supports {@code list} / {@code show} /
 * {@code rm} without a running server.
 *
 * <p>The manifest lives at {@code ~/.jllama/models.json} by default, overridable via the
 * {@value #PROP_PATH} system property. The in-memory map is keyed by name; an entry is also
 * resolvable through any of its {@link ModelRegistryEntry#getAliases() aliases}.
 *
 * <p>Pure Java (no native dependencies); persistence uses {@link java.nio.file.Files}. Mutating
 * operations are synchronized and persist immediately.
 */
public final class ModelRegistry {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** System property overriding the manifest file path. */
    public static final String PROP_PATH = "net.ladenthin.llama.registry.path";

    /** Default manifest location: {@code <user.home>/.jllama/models.json}. */
    public static final Path DEFAULT_PATH = Paths.get(System.getProperty("user.home", "."), ".jllama", "models.json");

    private final Path file;
    private final Map<String, ModelRegistryEntry> entries = new LinkedHashMap<>();

    /**
     * Load (or initialise) the registry at {@code file}. A missing file yields an empty registry.
     *
     * @param file the manifest path
     * @throws IOException if the file exists but cannot be read or parsed
     */
    public ModelRegistry(Path file) throws IOException {
        this.file = file;
        if (Files.exists(file)) {
            byte[] bytes = Files.readAllBytes(file);
            JsonNode root = MAPPER.readTree(new String(bytes, StandardCharsets.UTF_8));
            JsonNode models = root.path("models");
            if (models.isArray()) {
                for (JsonNode node : models) {
                    ModelRegistryEntry entry = ModelRegistryEntry.fromJson(node);
                    entries.put(entry.getName(), entry);
                }
            }
        }
    }

    /**
     * Load the registry using the system property {@value #PROP_PATH} if set, otherwise
     * {@link #DEFAULT_PATH}.
     *
     * @return the loaded registry
     * @throws IOException if the resolved file exists but cannot be read or parsed
     */
    public static ModelRegistry load() throws IOException {
        String override = System.getProperty(PROP_PATH);
        return new ModelRegistry(override != null && !override.isEmpty() ? Paths.get(override) : DEFAULT_PATH);
    }

    /**
     * Returns the manifest file path.
     *
     * @return the manifest file path
     */
    public Path getFile() {
        return file;
    }

    /**
     * Checks whether a model name or alias is registered.
     *
     * @param name the model name or alias
     * @return {@code true} if a model with this name (or alias) is registered
     */
    public synchronized boolean contains(String name) {
        return lookup(name) != null;
    }

    /**
     * Resolve a name or alias to its entry.
     *
     * @param name the model name or alias
     * @return the entry, or {@code null} when unknown
     */
    public synchronized @Nullable ModelRegistryEntry get(String name) {
        return lookup(name);
    }

    /**
     * Returns all registered entries in insertion order.
     *
     * @return all registered entries, in insertion order
     */
    public synchronized List<ModelRegistryEntry> list() {
        return new ArrayList<>(entries.values());
    }

    /**
     * Returns the number of registered models.
     *
     * @return the number of registered models
     */
    public synchronized int size() {
        return entries.size();
    }

    /**
     * Register or replace a model entry and persist the manifest.
     *
     * @param entry the entry to add or replace
     * @throws IOException if the manifest cannot be written
     */
    public synchronized void add(ModelRegistryEntry entry) throws IOException {
        entries.put(entry.getName(), entry);
        save();
    }

    /**
     * Remove a model by name or alias and persist the manifest.
     *
     * @param name the model name or alias
     * @return {@code true} if an entry was removed
     * @throws IOException if the manifest cannot be written
     */
    public synchronized boolean remove(String name) throws IOException {
        ModelRegistryEntry found = lookup(name);
        if (found == null) {
            return false;
        }
        entries.remove(found.getName());
        save();
        return true;
    }

    /**
     * Returns the full manifest as raw JSON.
     *
     * @return the full manifest as JSON (raw passthrough)
     */
    public synchronized JsonNode asJson() {
        return toJsonNode();
    }

    private @Nullable ModelRegistryEntry lookup(String name) {
        ModelRegistryEntry direct = entries.get(name);
        if (direct != null) {
            return direct;
        }
        for (ModelRegistryEntry entry : entries.values()) {
            if (entry.getAliases().contains(name)) {
                return entry;
            }
        }
        return null;
    }

    private JsonNode toJsonNode() {
        com.fasterxml.jackson.databind.node.ObjectNode root = MAPPER.createObjectNode();
        com.fasterxml.jackson.databind.node.ArrayNode models = root.putArray("models");
        for (ModelRegistryEntry entry : entries.values()) {
            models.add(entry.toJsonNode());
        }
        return root;
    }

    /** Persist the manifest to disk, creating parent directories as needed. */
    private void save() throws IOException {
        Path parent = file.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        String json = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(toJsonNode());
        Files.write(file, json.getBytes(StandardCharsets.UTF_8));
    }
}
