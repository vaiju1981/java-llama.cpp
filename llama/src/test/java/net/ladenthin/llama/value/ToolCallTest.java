// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin ToolCall's id/name/argumentsJson accessors, its hand-written function-call toString "
                + "(name(args)[id]), and its Lombok value-equality so every mutation is covered.")
public class ToolCallTest {

    @Test
    public void accessorsReturnConstructorValues() {
        ToolCall tc = new ToolCall("call_1", "get_weather", "{\"city\":\"Berlin\"}");
        assertThat(tc.getId(), is("call_1"));
        assertThat(tc.getName(), is("get_weather"));
        assertThat(tc.getArgumentsJson(), is("{\"city\":\"Berlin\"}"));
    }

    @Test
    public void toStringRendersFunctionCallSyntax() {
        // Hand-written toString: name(argsJson)[id] — assert the exact string so the
        // empty-return mutant ("") and any field-omission mutant are killed.
        ToolCall tc = new ToolCall("c1", "add", "{\"a\":2}");
        assertThat(tc.toString(), is("add({\"a\":2})[c1]"));
    }

    @Test
    public void equalsAndHashCodeAreValueBased() {
        ToolCall a = new ToolCall("c1", "add", "{}");
        ToolCall b = new ToolCall("c1", "add", "{}");
        assertThat(a, is(b));
        assertThat(a.hashCode(), is(b.hashCode()));
    }

    @Test
    public void differingNameBreaksEquality() {
        assertThat(new ToolCall("c1", "add", "{}"), is(not(new ToolCall("c1", "sub", "{}"))));
    }
}
