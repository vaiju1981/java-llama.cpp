// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Model-free coverage of {@link ModelNameResolver}: URL/local/alias/HF-spec resolution. */
public class ModelNameResolverTest {

    @Test
    void resolvesDirectUrlAndDerivesNameAndQuant() {
        ModelNameResolver r = new ModelNameResolver();
        ResolvedModelSource s = r.resolve("https://example.com/models/foo-Q4_K_M.gguf");
        assertEquals("foo-Q4_K_M", s.getName());
        assertEquals("https://example.com/models/foo-Q4_K_M.gguf", s.getUrl());
        assertEquals("Q4_K_M", s.getQuantization());
        assertTrue(s.isRemote());
        assertFalse(s.isLocal());
    }

    @Test
    void resolvesLocalPathWithoutDownload() {
        ModelNameResolver r = new ModelNameResolver();
        ResolvedModelSource s = r.resolve("/tmp/my-model.gguf");
        assertEquals("my-model", s.getName());
        assertEquals("/tmp/my-model.gguf", s.getLocalPath());
        assertTrue(s.isLocal());
    }

    @Test
    void resolvesCuratedAlias() {
        ModelNameResolver r = new ModelNameResolver(Map.of("llama3.2", "https://hf.co/m/llama3.2-Q4_K_M.gguf"));
        ResolvedModelSource s = r.resolve("llama3.2");
        assertEquals("llama3.2-Q4_K_M", s.getName());
        assertEquals("https://hf.co/m/llama3.2-Q4_K_M.gguf", s.getUrl());
        assertEquals("Q4_K_M", s.getQuantization());
    }

    @Test
    void quantOverrideOnAlias() {
        ModelNameResolver r = new ModelNameResolver(Map.of("llama3.2", "https://hf.co/m/llama3.2.gguf"));
        ResolvedModelSource s = r.resolve("llama3.2@Q8_0");
        assertEquals("llama3.2", s.getName());
        assertEquals("Q8_0", s.getQuantization());
    }

    @Test
    void unknownShortNameThrows() {
        ModelNameResolver r = new ModelNameResolver(Map.of("known", "https://x/y.gguf"));
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> r.resolve("totally-unknown-model"));
        assertTrue(ex.getMessage().contains("Unknown model"));
    }

    @Test
    void blankSpecThrows() {
        ModelNameResolver r = new ModelNameResolver();
        assertThrows(IllegalArgumentException.class, () -> r.resolve("   "));
    }

    @Test
    void hfTokenAddsAuthHeader() {
        ModelNameResolver r = new ModelNameResolver(Map.of(), "secret-token");
        ResolvedModelSource s = r.resolve("https://huggingface.co/org/repo/resolve/main/m.gguf");
        assertEquals("Bearer secret-token", s.getHeaders().get("Authorization"));
    }

    @Test
    void bestEffortHfSpec() {
        ModelNameResolver r = new ModelNameResolver();
        ResolvedModelSource s = r.resolve("org/my-repo@F16");
        assertTrue(s.getUrl().startsWith("https://huggingface.co/org/my-repo/resolve/main/my-repo.gguf"));
        assertEquals("F16", s.getQuantization());
        assertNull(s.getHeaders().get("Authorization"));
    }
}
