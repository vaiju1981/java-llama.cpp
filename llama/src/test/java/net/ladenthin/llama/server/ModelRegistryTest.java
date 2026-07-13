// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Model-free coverage of {@link ModelRegistry}: add/remove/lookup, aliases, persistence, JSON. */
public class ModelRegistryTest {

    @TempDir
    Path tempDir;

    private ModelRegistry newRegistry() throws Exception {
        return new ModelRegistry(tempDir.resolve("models.json"));
    }

    private ModelRegistryEntry sample(String name) {
        return new ModelRegistryEntry.Builder(name)
                .localPath("models/" + name + ".gguf")
                .sourceUrl("https://hf.co/org/" + name + ".gguf")
                .quantization("Q4_K_M")
                .sizeBytes(1234L)
                .aliases(java.util.Arrays.asList(name + ":latest"))
                .pulledAt(1700000000000L)
                .build();
    }

    @Test
    void emptyRegistryHasNoEntries() throws Exception {
        ModelRegistry registry = newRegistry();
        assertEquals(0, registry.size());
        assertNull(registry.get("anything"));
        assertFalse(registry.contains("anything"));
    }

    @Test
    void addThenLookupByNameAndAlias() throws Exception {
        ModelRegistry registry = newRegistry();
        registry.add(sample("llama3.2"));
        assertTrue(registry.contains("llama3.2"));
        assertEquals("models/llama3.2.gguf", registry.get("llama3.2").getLocalPath());
        // resolves via alias too
        assertTrue(registry.contains("llama3.2:latest"));
        assertEquals("llama3.2", registry.get("llama3.2:latest").getName());
    }

    @Test
    void removeByNameAndByAlias() throws Exception {
        ModelRegistry registry = newRegistry();
        registry.add(sample("llama3.2"));
        assertTrue(registry.remove("llama3.2:latest"));
        assertEquals(0, registry.size());
        assertFalse(registry.contains("llama3.2"));
        assertFalse(registry.remove("missing"));
    }

    @Test
    void persistsToDiskAndReloads() throws Exception {
        Path file = tempDir.resolve("models.json");
        ModelRegistry first = new ModelRegistry(file);
        first.add(sample("qwen2.5"));
        assertTrue(Files.exists(file));

        // A fresh registry reading the same file sees the entry.
        ModelRegistry reloaded = new ModelRegistry(file);
        assertEquals(1, reloaded.size());
        assertEquals("Q4_K_M", reloaded.get("qwen2.5").getQuantization());
    }

    @Test
    void asJsonReflectsEntries() throws Exception {
        ModelRegistry registry = newRegistry();
        registry.add(sample("phi3"));
        JsonNode json = registry.asJson();
        assertTrue(json.path("models").isArray());
        assertEquals(1, json.path("models").size());
        assertEquals("phi3", json.path("models").get(0).path("name").asText());
        assertNotNull(json.path("models").get(0).path("pulled_at").asText());
    }
}
