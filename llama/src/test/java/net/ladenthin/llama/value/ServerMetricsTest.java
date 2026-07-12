// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify ServerMetrics typed getters map all fields emitted by server_task_result_metrics::to_json, "
                + "including cumulative Usage and derived cumulative Timings.")
public class ServerMetricsTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ServerMetrics parse(String json) throws Exception {
        return new ServerMetrics(MAPPER.readTree(json));
    }

    private static final String SAMPLE = "{\"idle\":2,\"processing\":1,\"deferred\":3,\"t_start\":1234567890,"
            + "\"n_prompt_tokens_processed_total\":100,\"t_prompt_processing_total\":50,"
            + "\"n_tokens_predicted_total\":200,\"t_tokens_generation_total\":80,"
            + "\"n_prompt_tokens_processed\":10,\"t_prompt_processing\":5,"
            + "\"n_tokens_predicted\":20,\"t_tokens_generation\":8,"
            + "\"n_decode_total\":300,\"n_busy_slots_total\":4,\"n_tokens_max\":4096,"
            // next_token is an ARRAY of one object — this mirrors llama.cpp's server_slot::to_json
            // at b9739, not a bare object; SlotMetrics must unwrap next_token[0].
            + "\"slots\":[{\"id\":0,\"n_ctx\":4096,\"is_processing\":true,"
            + "\"n_prompt_tokens\":100,\"n_prompt_tokens_processed\":20,"
            + "\"n_prompt_tokens_cache\":80,\"next_token\":[{\"n_decoded\":7,\"n_remain\":9}]},"
            + "{\"id\":1}]}";

    @Test
    public void slotCountsAndTimestamp() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(2, m.getIdleSlots());
        assertEquals(1, m.getProcessingSlots());
        assertEquals(3, m.getDeferredTasks());
        assertEquals(1234567890L, m.getStartTimestamp());
    }

    @Test
    public void totalsAndMaxTokens() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(300L, m.getDecodeTotal());
        assertEquals(4L, m.getBusySlotsTotal());
        assertEquals(4096, m.getTokensMax());
    }

    @Test
    public void cumulativeUsage() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Usage u = m.getCumulativeUsage();
        assertEquals(100L, u.getPromptTokens());
        assertEquals(200L, u.getCompletionTokens());
        assertEquals(300L, u.getTotalTokens());
        assertEquals(100L, m.getCumulativeProcessedPromptTokens());
        assertEquals(200L, m.getCumulativeGeneratedTokens());
    }

    @Test
    public void windowUsage() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Usage u = m.getWindowUsage();
        assertEquals(10L, u.getPromptTokens());
        assertEquals(20L, u.getCompletionTokens());
        assertEquals(10L, m.getWindowProcessedPromptTokens());
    }

    @Test
    public void cumulativeTimingsDerivesRates() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Timings t = m.getCumulativeTimings();
        assertEquals(100, t.getPromptN());
        assertEquals(50.0, t.getPromptMs(), 1e-9);
        // 100 tokens / 50ms = 2000/s
        assertEquals(2000.0, t.getPromptPerSecond(), 1e-9);
        // 200 / 80ms = 2500/s
        assertEquals(2500.0, t.getPredictedPerSecond(), 1e-9);
    }

    @Test
    public void cumulativeTimingsDoNotOverflowIntOnBusyServer() throws Exception {
        // A long-running server easily exceeds Integer.MAX_VALUE tokens; an int cast would wrap
        // these cumulative counters to negative values.
        ServerMetrics m =
                parse("{\"n_prompt_tokens_processed_total\":3000000000," + "\"n_tokens_predicted_total\":4000000000}");
        Timings t = m.getCumulativeTimings();
        assertEquals(3_000_000_000L, t.getPromptN());
        assertEquals(4_000_000_000L, t.getPredictedN());
    }

    @Test
    public void cumulativeTimingsZeroMsYieldsZeroRate() throws Exception {
        ServerMetrics m = parse("{\"n_prompt_tokens_processed_total\":5,\"t_prompt_processing_total\":0}");
        assertEquals(0.0, m.getCumulativeTimings().getPromptPerSecond(), 1e-9);
    }

    @Test
    public void slotsArrayExposed() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertTrue(m.getSlots().isArray());
        assertEquals(2, m.getSlots().size());
    }

    @Test
    public void typedSlotMetricsExposeCacheCounts() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(2, m.getSlotMetrics().size());
        SlotMetrics slot = m.getSlotMetrics().get(0);
        assertEquals(0, slot.getId());
        assertEquals(4096, slot.getContextSize());
        assertTrue(slot.isProcessing());
        assertEquals(100L, slot.getPromptTokens());
        assertEquals(20L, slot.getProcessedPromptTokens());
        assertEquals(80L, slot.getCachedPromptTokens());
        assertEquals(7L, slot.getDecodedTokens());
        assertEquals(9L, slot.getRemainingTokens());
        assertEquals(0, slot.asJson().path("id").asInt());
        assertTrue(slot.toString().contains("n_prompt_tokens_cache"));

        // Assert against the SECOND slot (id=1, no is_processing) so the id and is_processing
        // accessors are pinned to non-default values a constant-return mutant cannot satisfy:
        // slot 0's id==0 and is_processing==true coincide with the mutated constants.
        SlotMetrics idle = m.getSlotMetrics().get(1);
        assertEquals(1, idle.getId());
        assertFalse(idle.isProcessing());
        // next_token absent on the idle slot — accessors fall back to zero, not throw.
        assertEquals(0L, idle.getDecodedTokens());
        assertEquals(0L, idle.getRemainingTokens());
    }

    @Test
    public void missingFieldsDefaultToZero() throws Exception {
        ServerMetrics m = parse("{}");
        assertEquals(0, m.getIdleSlots());
        assertEquals(0L, m.getDecodeTotal());
        assertEquals(0, m.getTokensMax());
        assertEquals(0L, m.getCumulativeUsage().getTotalTokens());
    }

    @Test
    public void cumulativeTimingsZeroPredictedMsYieldsZeroRate() throws Exception {
        // Pins the predictedMs > 0.0 boundary: with predictedN>0 but predictedMs=0 the rate must be 0.0
        // (a >= boundary mutant would divide by zero and produce a non-zero / NaN rate).
        ServerMetrics m = parse("{\"n_tokens_predicted_total\":5,\"t_tokens_generation_total\":0}");
        assertEquals(0.0, m.getCumulativeTimings().getPredictedPerSecond(), 1e-9);
    }

    @Test
    public void asJsonExposesBackingNode() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // Dereferencing the returned node kills the "return null" mutant on asJson().
        assertEquals(2, m.asJson().get("idle").asInt());
    }

    @Test
    public void toStringSerializesNode() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // Assert content (not just non-null) so the empty-string return mutant on toString is killed.
        assertTrue(m.toString().contains("idle"));
    }

    @Test
    public void queueDepthMirrorsDeferred() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(3, m.getQueueDepth());
    }

    @Test
    public void promptCacheHitsAggregatedAcrossSlots() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // slot 0 has n_prompt_tokens_cache=80, slot 1 has none.
        assertEquals(80L, m.getPromptCacheHits());
    }

    @Test
    public void kvCacheUsageDerivedFromSlots() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // used = sum(n_prompt_tokens) = 100 (slot 0) + 0 (slot 1); capacity = sum(n_ctx) = 4096.
        assertEquals(100L, m.getKvCacheUsedTokens());
        assertEquals(4096L, m.getKvCacheCapacityTokens());
        assertEquals(100.0 / 4096.0, m.getKvCacheUsage(), 1e-9);
    }

    @Test
    public void devicesAbsentOnCurrentShape() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(0, m.getDeviceCount());
        assertFalse(m.getDevices().isArray());
    }

    @Test
    public void slotCacheHitRateComputed() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        SlotMetrics slot = m.getSlotMetrics().get(0);
        // cached=80 / (cached+processed=100) = 0.8
        assertEquals(0.8, slot.getCacheHitRate(), 1e-9);
    }

    @Test
    public void deviceUtilizationAndVramAbsentOnCurrentShape() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // SAMPLE has no devices array: getters degrade to zero rather than throwing.
        assertEquals(0.0, m.getDeviceUtilization(0), 1e-9);
        assertEquals(0L, m.getVramUsedBytes(0));
        assertEquals(0.0, m.getDeviceUtilization(-1), 1e-9);
        assertEquals(0L, m.getVramUsedBytes(99));
    }

    @Test
    public void deviceUtilizationAndVramParsedWhenPresent() throws Exception {
        ObjectNode devices =
                MAPPER.createArrayNode().addObject().put("utilization", 42.5).put("vram_used", 1_073_741_824L);
        ObjectNode root = (ObjectNode) MAPPER.readTree(SAMPLE);
        root.set("devices", devices);
        ServerMetrics m = new ServerMetrics(root);
        assertEquals(42.5, m.getDeviceUtilization(0), 1e-9);
        assertEquals(1_073_741_824L, m.getVramUsedBytes(0));
    }
}
