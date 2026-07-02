// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

public class ContinuationModeTest {

    @Test
    public void getValueReturnsWireFormatStrings() {
        // Pinning the exact wire strings kills the empty-string return mutant on getValue().
        assertThat(ContinuationMode.REASONING_CONTENT.getValue(), is("reasoning_content"));
        assertThat(ContinuationMode.CONTENT.getValue(), is("content"));
    }
}
