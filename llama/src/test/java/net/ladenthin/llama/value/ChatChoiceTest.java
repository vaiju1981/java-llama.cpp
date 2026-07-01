// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin every ChatChoice accessor to a distinct non-default value so the index/message/"
                + "finishReason getters and the value-semantics of equals/hashCode/toString are all "
                + "mutation-covered.")
public class ChatChoiceTest {

    private static ChatChoice choice(int index, String role, String content, String finish) {
        return new ChatChoice(index, new ChatMessage(role, content), finish);
    }

    @Test
    public void accessorsReturnConstructorValues() {
        ChatMessage msg = new ChatMessage("assistant", "hello");
        ChatChoice c = new ChatChoice(7, msg, "stop");
        // index getter — a non-zero value kills the "return 0" primitive mutant.
        assertThat(c.getIndex(), is(7));
        assertThat(c.getMessage(), is(sameInstance(msg)));
        assertThat(c.getFinishReason(), is("stop"));
    }

    @Test
    public void toStringRendersAllFields() {
        ChatChoice c = choice(3, "assistant", "hi there", "length");
        String s = c.toString();
        assertThat(s, containsString("3"));
        assertThat(s, containsString("length"));
        assertThat(s, containsString("hi there"));
    }

    @Test
    public void equalsAndHashCodeAreValueBased() {
        ChatChoice a = choice(1, "assistant", "x", "stop");
        ChatChoice b = choice(1, "assistant", "x", "stop");
        assertThat(a, is(b));
        assertThat(a.hashCode(), is(b.hashCode()));
    }

    @Test
    public void differingIndexBreaksEquality() {
        assertThat(choice(1, "assistant", "x", "stop"), is(not(choice(2, "assistant", "x", "stop"))));
    }

    @Test
    public void differingFinishReasonBreaksEquality() {
        assertThat(choice(1, "assistant", "x", "stop"), is(not(choice(1, "assistant", "x", "length"))));
    }
}
