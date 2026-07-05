// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the LoraAdapter value type: constructor/getter round-trip for every "
                + "field, and the Lombok-generated equals/hashCode/toString contracts used when "
                + "adapters are compared or logged.")
public class LoraAdapterTest {

    private static LoraAdapter sample() {
        return new LoraAdapter(1, "adapter.gguf", 0.5f, "classification", "prefix: ");
    }

    @Test
    public void gettersRoundTrip() {
        LoraAdapter adapter = sample();
        assertThat(adapter.getId(), is(1));
        assertThat(adapter.getPath(), is("adapter.gguf"));
        assertEquals(0.5f, adapter.getScale(), 0.0f);
        assertThat(adapter.getTaskName(), is("classification"));
        assertThat(adapter.getPromptPrefix(), is("prefix: "));
    }

    @Test
    public void equalsAndHashCode_sameValues() {
        assertEquals(sample(), sample());
        assertEquals(sample().hashCode(), sample().hashCode());
    }

    @Test
    public void equals_differsPerField() {
        LoraAdapter base = sample();
        assertNotEquals(base, new LoraAdapter(2, "adapter.gguf", 0.5f, "classification", "prefix: "));
        assertNotEquals(base, new LoraAdapter(1, "other.gguf", 0.5f, "classification", "prefix: "));
        assertNotEquals(base, new LoraAdapter(1, "adapter.gguf", 1.0f, "classification", "prefix: "));
        assertNotEquals(base, new LoraAdapter(1, "adapter.gguf", 0.5f, "other", "prefix: "));
        assertNotEquals(base, new LoraAdapter(1, "adapter.gguf", 0.5f, "classification", "other"));
    }

    @Test
    public void toStringContainsFields() {
        String rendered = sample().toString();
        assertThat(rendered.contains("adapter.gguf"), is(true));
        assertThat(rendered.contains("classification"), is(true));
        assertThat(rendered, is(not("")));
    }
}
