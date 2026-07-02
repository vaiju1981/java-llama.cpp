// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/** Model-free tests for the {@link Optimizer} native-value mapping. */
class OptimizerTest {

    @Test
    void nativeValuesMatchGgml() {
        assertThat(Optimizer.ADAMW.getNativeValue(), is(0));
        assertThat(Optimizer.SGD.getNativeValue(), is(1));
    }
}
