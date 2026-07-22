// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Model-free coverage of {@link OllamaRegistryCompat}: jllama registry → Ollama /api/tags shape. */
public class OllamaRegistryCompatTest {

    @TempDir
    Path tempDir;

    private ModelRegistry sample() throws IOException {
        ModelRegistry registry = new ModelRegistry(tempDir.resolve("models.json"));
        registry.add(new ModelRegistryEntry.Builder("llama3.2")
                .localPath("/models/llama3.2.gguf")
                .quantization("Q4_K_M")
                .sizeBytes(1234L)
                .pulledAt(1_700_000_000_000L)
                .build());
        registry.add(new ModelRegistryEntry.Builder("qwen2.5")
                .sourceUrl("https://hf.co/q/qwen2.5.gguf")
                .sizeBytes(0L)
                .pulledAt(0L)
                .build());
        return registry;
    }

    @Test
    void listMirrorsOllamaTagsShape() throws Exception {
        ObjectMapper m = new ObjectMapper();
        JsonNode root = m.readTree(OllamaRegistryCompat.listAsOllamaTags(sample()));
        assertTrue(root.path("models").isArray());
        assertEquals(2, root.path("models").size());

        JsonNode first = root.path("models").get(0);
        assertEquals("llama3.2", first.path("name").asText());
        assertEquals("llama3.2", first.path("model").asText());
        assertEquals(1234L, first.path("size").asLong());
        assertEquals("Q4_K_M", first.path("details").path("quantization_level").asText());
        assertFalse(first.path("digest").asText().isEmpty());
        assertTrue(first.path("modified_at").asText().contains("2023"));

        JsonNode second = root.path("models").get(1);
        assertEquals("qwen2.5", second.path("name").asText());
        // missing pulledAt → empty modified_at; missing quant → empty quantization_level
        assertEquals("", second.path("modified_at").asText());
        assertEquals("", second.path("details").path("quantization_level").asText());
    }

    @Test
    void singleEntryShape() throws Exception {
        ObjectMapper m = new ObjectMapper();
        ModelRegistry registry = new ModelRegistry(tempDir.resolve("m.json"));
        registry.add(new ModelRegistryEntry.Builder("x")
                .localPath("/x.gguf")
                .quantization("F16")
                .sizeBytes(9L)
                .pulledAt(5L)
                .build());
        JsonNode node = m.readTree(
                OllamaRegistryCompat.entryAsOllamaTag(registry.get("x")).toString());
        assertEquals("x", node.path("name").asText());
        assertEquals("F16", node.path("details").path("quantization_level").asText());
    }
}
