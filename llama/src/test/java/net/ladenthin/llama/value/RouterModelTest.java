// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the RouterModel value type: constructor/getter round-trip, the "
                + "Status.fromValue mapping for every upstream server_model_status string "
                + "(plus null/unknown tolerance), the Lombok equals/hashCode contract, and "
                + "the handwritten toString shapes used in router log traces.")
public class RouterModelTest {

    private static RouterModel sample() {
        return new RouterModel("Qwen3-0.6B-Q4_K_M", RouterModel.Status.LOADED, "loaded", false, 0);
    }

    @Test
    public void gettersRoundTrip() {
        RouterModel model = sample();
        assertThat(model.getId(), is("Qwen3-0.6B-Q4_K_M"));
        assertThat(model.getStatus(), is(RouterModel.Status.LOADED));
        assertThat(model.getStatusValue(), is("loaded"));
        assertThat(model.isFailed(), is(false));
        assertThat(model.getExitCode(), is(0));
    }

    @Test
    public void statusFromValue_mapsEveryUpstreamString() {
        // The exact strings emitted by upstream server_model_status_to_string (server-models.h).
        assertThat(RouterModel.Status.fromValue("downloading"), is(RouterModel.Status.DOWNLOADING));
        assertThat(RouterModel.Status.fromValue("downloaded"), is(RouterModel.Status.DOWNLOADED));
        assertThat(RouterModel.Status.fromValue("unloaded"), is(RouterModel.Status.UNLOADED));
        assertThat(RouterModel.Status.fromValue("loading"), is(RouterModel.Status.LOADING));
        assertThat(RouterModel.Status.fromValue("loaded"), is(RouterModel.Status.LOADED));
        assertThat(RouterModel.Status.fromValue("sleeping"), is(RouterModel.Status.SLEEPING));
    }

    @Test
    public void statusFromValue_matchesExactlyNotCaseFolded() {
        // Upstream emits exactly lowercase strings; matching is deliberately exact
        // (no Unicode case transformation — findsecbugs IMPROPER_UNICODE).
        assertThat(RouterModel.Status.fromValue("LOADED"), is(RouterModel.Status.UNKNOWN));
    }

    @Test
    public void statusFromValue_toleratesNullAndUnrecognized() {
        assertThat(RouterModel.Status.fromValue(null), is(RouterModel.Status.UNKNOWN));
        assertThat(RouterModel.Status.fromValue(""), is(RouterModel.Status.UNKNOWN));
        assertThat(RouterModel.Status.fromValue("hibernating"), is(RouterModel.Status.UNKNOWN));
    }

    @Test
    public void equalsAndHashCode_sameValues() {
        assertEquals(sample(), sample());
        assertEquals(sample().hashCode(), sample().hashCode());
    }

    @Test
    public void equals_differsPerField() {
        RouterModel base = sample();
        assertNotEquals(base, new RouterModel("other", RouterModel.Status.LOADED, "loaded", false, 0));
        assertNotEquals(base, new RouterModel("Qwen3-0.6B-Q4_K_M", RouterModel.Status.LOADING, "loaded", false, 0));
        assertNotEquals(base, new RouterModel("Qwen3-0.6B-Q4_K_M", RouterModel.Status.LOADED, "loading", false, 0));
        assertNotEquals(base, new RouterModel("Qwen3-0.6B-Q4_K_M", RouterModel.Status.LOADED, "loaded", true, 0));
        assertNotEquals(base, new RouterModel("Qwen3-0.6B-Q4_K_M", RouterModel.Status.LOADED, "loaded", false, 1));
    }

    @Test
    public void toString_healthyShape() {
        assertThat(sample().toString(), is("Qwen3-0.6B-Q4_K_M [loaded]"));
    }

    @Test
    public void toString_failedShapeIncludesExitCode() {
        RouterModel failed = new RouterModel("broken", RouterModel.Status.UNLOADED, "unloaded", true, 137);
        assertThat(failed.toString(), is("broken [unloaded, failed exit=137]"));
    }
}
