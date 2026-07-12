// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayWithSize;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.args.GpuSplitMode;
import net.ladenthin.llama.args.MiroStat;
import net.ladenthin.llama.args.NumaStrategy;
import net.ladenthin.llama.args.PoolingType;
import net.ladenthin.llama.args.RopeScalingType;
import net.ladenthin.llama.args.Sampler;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify ModelParameters input validation (priority 0-3, repeatLastN/dryPenaltyLastN >= -1), "
                + "correct CLI argument formatting for enum-based setters (PoolingType, RopeScalingType, "
                + "CacheType, GpuSplitMode, NumaStrategy, MiroStat) and composite-value setters "
                + "(loraScaled, controlVectorScaled, controlVectorLayerRange), semicolon-separated "
                + "lowercase sampler list, isUnset key-presence check, and the CliParameters base "
                + "behaviour: toString omits 'null' for flag-only entries, toArray always prepends an "
                + "empty argv[0] string and omits values for null-valued flags.")
public class ModelParametersTest {

    // -------------------------------------------------------------------------
    // setPriority — validation (0-3 only)
    // -------------------------------------------------------------------------

    @Test
    public void testSetPriorityValid0() {
        ModelParameters p = new ModelParameters().setPriority(0);
        assertThat(p.parameters.get("--prio"), is("0"));
    }

    @Test
    public void testSetPriorityValid3() {
        ModelParameters p = new ModelParameters().setPriority(3);
        assertThat(p.parameters.get("--prio"), is("3"));
    }

    @Test
    public void testSetPriorityNegative() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriority(-1));
    }

    @Test
    public void testSetPriorityTooHigh() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriority(4));
    }

    // -------------------------------------------------------------------------
    // setPriorityBatch — validation (0-3 only)
    // -------------------------------------------------------------------------

    @Test
    public void testSetPriorityBatchValid1() {
        ModelParameters p = new ModelParameters().setPriorityBatch(1);
        assertThat(p.parameters.get("--prio-batch"), is("1"));
    }

    @Test
    public void testSetPriorityBatchNegative() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriorityBatch(-1));
    }

    @Test
    public void testSetPriorityBatchTooHigh() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriorityBatch(4));
    }

    // -------------------------------------------------------------------------
    // setRepeatLastN — validation (>= -1)
    // -------------------------------------------------------------------------

    @Test
    public void testSetRepeatLastNValidZero() {
        ModelParameters p = new ModelParameters().setRepeatLastN(0);
        assertThat(p.parameters.get("--repeat-last-n"), is("0"));
    }

    @Test
    public void testSetRepeatLastNValidMinusOne() {
        ModelParameters p = new ModelParameters().setRepeatLastN(-1);
        assertThat(p.parameters.get("--repeat-last-n"), is("-1"));
    }

    @Test
    public void testSetRepeatLastNValid64() {
        ModelParameters p = new ModelParameters().setRepeatLastN(64);
        assertThat(p.parameters.get("--repeat-last-n"), is("64"));
    }

    @Test
    public void testSetRepeatLastNTooLow() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setRepeatLastN(-2));
    }

    // -------------------------------------------------------------------------
    // setDryPenaltyLastN — validation (>= -1)
    // -------------------------------------------------------------------------

    @Test
    public void testSetDryPenaltyLastNValidMinusOne() {
        ModelParameters p = new ModelParameters().setDryPenaltyLastN(-1);
        assertThat(p.parameters.get("--dry-penalty-last-n"), is("-1"));
    }

    @Test
    public void testSetDryPenaltyLastNValidZero() {
        ModelParameters p = new ModelParameters().setDryPenaltyLastN(0);
        assertThat(p.parameters.get("--dry-penalty-last-n"), is("0"));
    }

    @Test
    public void testSetDryPenaltyLastNTooLow() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setDryPenaltyLastN(-2));
    }

    // -------------------------------------------------------------------------
    // setSamplers — semicolon-separated lowercase names
    // -------------------------------------------------------------------------

    @Test
    public void testSetSamplersSingle() {
        ModelParameters p = new ModelParameters().setSamplers(Sampler.TOP_K);
        assertThat(p.parameters.get("--samplers"), is("top_k"));
    }

    @Test
    public void testSetSamplersMultiple() {
        ModelParameters p = new ModelParameters().setSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);
        assertThat(p.parameters.get("--samplers"), is("top_k;top_p;temperature"));
    }

    @Test
    public void testSetSamplersEmpty() {
        ModelParameters p = new ModelParameters().setSamplers();
        assertThat(p.parameters, not(hasKey("--samplers")));
    }

    @Test
    public void testSetSamplersAllLowercase() {
        for (Sampler s : Sampler.values()) {
            ModelParameters p = new ModelParameters().setSamplers(s);
            assertThat(p.parameters.get("--samplers"), is(s.name().toLowerCase()));
        }
    }

    // -------------------------------------------------------------------------
    // addLoraScaledAdapter / addControlVectorScaled — "fname,scale" format
    // -------------------------------------------------------------------------

    @Test
    public void testAddLoraScaledAdapter() {
        ModelParameters p = new ModelParameters().addLoraScaledAdapter("adapter.bin", 0.5f);
        assertThat(p.parameters.get("--lora-scaled"), is("adapter.bin,0.5"));
    }

    @Test
    public void testAddControlVectorScaled() {
        ModelParameters p = new ModelParameters().addControlVectorScaled("vec.bin", 1.5f);
        assertThat(p.parameters.get("--control-vector-scaled"), is("vec.bin,1.5"));
    }

    // -------------------------------------------------------------------------
    // setControlVectorLayerRange — "start,end" format
    // -------------------------------------------------------------------------

    @Test
    public void testSetControlVectorLayerRange() {
        ModelParameters p = new ModelParameters().setControlVectorLayerRange(2, 10);
        assertThat(p.parameters.get("--control-vector-layer-range"), is("2,10"));
    }

    @Test
    public void testSetControlVectorLayerRangeSameStartEnd() {
        ModelParameters p = new ModelParameters().setControlVectorLayerRange(5, 5);
        assertThat(p.parameters.get("--control-vector-layer-range"), is("5,5"));
    }

    // -------------------------------------------------------------------------
    // isUnset
    // -------------------------------------------------------------------------

    @Test
    public void testIsDefaultTrueWhenNotSet() {
        ModelParameters p = new ModelParameters();
        assertThat(p.isUnset("threads"), is(true));
    }

    @Test
    public void testIsDefaultFalseWhenSet() {
        ModelParameters p = new ModelParameters().setThreads(4);
        assertThat(p.isUnset("threads"), is(false));
    }

    @Test
    public void testIsDefaultFalseAfterFlagOnly() {
        ModelParameters p = new ModelParameters().enableEmbedding();
        assertThat(p.isUnset("embedding"), is(false));
    }

    // -------------------------------------------------------------------------
    // Enum-based setters (PoolingType, RopeScalingType, CacheType, etc.)
    // -------------------------------------------------------------------------

    @Test
    public void testSetPoolingTypeMean() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.MEAN);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.MEAN.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeNone() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.NONE);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.NONE.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeCls() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.CLS);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.CLS.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeLast() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.LAST);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.LAST.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeRank() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.RANK);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.RANK.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeUnspecifiedDoesNotSetParam() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.UNSPECIFIED);
        assertThat(
                "UNSPECIFIED pooling type must not add " + ModelParameters.ARG_POOLING + " to parameters",
                p.parameters,
                not(hasKey(ModelParameters.ARG_POOLING)));
    }

    @Test
    public void testSetPoolingTypeUnspecifiedLeavesDefaultUntouched() {
        // A fresh ModelParameters must not have ARG_POOLING set by default either
        ModelParameters fresh = new ModelParameters();
        assertThat(fresh.parameters, not(hasKey(ModelParameters.ARG_POOLING)));
        // Calling setPoolingType(UNSPECIFIED) must leave that invariant intact
        fresh.setPoolingType(PoolingType.UNSPECIFIED);
        assertThat(fresh.parameters, not(hasKey(ModelParameters.ARG_POOLING)));
    }

    @Test
    public void testSetRopeScaling() {
        ModelParameters p = new ModelParameters().setRopeScaling(RopeScalingType.YARN2);
        assertThat(p.parameters.get("--rope-scaling"), is("yarn"));
    }

    @Test
    public void testSetCacheTypeKLowercase() {
        ModelParameters p = new ModelParameters().setCacheTypeK(CacheType.F16);
        assertThat(p.parameters.get("--cache-type-k"), is("f16"));
    }

    @Test
    public void testSetCacheTypeVLowercase() {
        ModelParameters p = new ModelParameters().setCacheTypeV(CacheType.Q8_0);
        assertThat(p.parameters.get("--cache-type-v"), is("q8_0"));
    }

    @Test
    public void testSetSplitModeLowercase() {
        ModelParameters p = new ModelParameters().setSplitMode(GpuSplitMode.LAYER);
        assertThat(p.parameters.get("--split-mode"), is("layer"));
    }

    @Test
    public void testSetNumaLowercase() {
        ModelParameters p = new ModelParameters().setNuma(NumaStrategy.DISTRIBUTE);
        assertThat(p.parameters.get("--numa"), is("distribute"));
    }

    @Test
    public void testSetMirostatOrdinal() {
        ModelParameters p = new ModelParameters().setMirostat(MiroStat.V2);
        assertThat(p.parameters.get("--mirostat"), is("2"));
    }

    // -------------------------------------------------------------------------
    // CliParameters.toString() — space-separated key[space value] pairs
    // -------------------------------------------------------------------------

    @Test
    public void testToStringContainsKey() {
        ModelParameters p = new ModelParameters().setThreads(4);
        assertThat(p.toString(), containsString("--threads"));
        assertThat(p.toString(), containsString("4"));
    }

    @Test
    public void testToStringFlagOnlyNoValue() {
        ModelParameters p = new ModelParameters().enableEmbedding();
        String s = p.toString();
        assertThat(s, containsString("--embedding"));
        // Flag-only: value is null, so no "null" text should appear
        assertThat(s, not(containsString("null")));
    }

    @Test
    public void testFitValueTrueReturnsFitOn() {
        assertThat(ModelParameters.fitValue(true), is(ModelParameters.FIT_ON));
    }

    @Test
    public void testFitValueFalseReturnsFitOff() {
        assertThat(ModelParameters.fitValue(false), is(ModelParameters.FIT_OFF));
    }

    @Test
    public void testToStringDefaultContainsFit() {
        ModelParameters p = new ModelParameters();
        String s = p.toString();
        assertThat(s, containsString("--fit"));
        assertThat(s, containsString(ModelParameters.DEFAULT_FIT_VALUE));
    }

    // -------------------------------------------------------------------------
    // CliParameters.toArray() — leading empty string + key/value pairs
    // -------------------------------------------------------------------------

    @Test
    public void testToArrayDefaultParametersHasFit() {
        // toArray() = ["", "--fit", DEFAULT_FIT_VALUE]
        ModelParameters p = new ModelParameters();
        String[] arr = p.toArray();
        assertThat(arr, arrayWithSize(3));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    @Test
    public void testToArrayScalarParameterHasFiveElements() {
        // argv[0]="" + "--fit" + DEFAULT_FIT_VALUE + "--threads" + "4" = 5
        ModelParameters p = new ModelParameters().setThreads(4);
        String[] arr = p.toArray();
        assertThat(arr, arrayWithSize(5));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--threads"));
        assertThat(list, hasItem("4"));
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    @Test
    public void testToArrayFlagOnlyHasFourElements() {
        // argv[0]="" + "--fit" + DEFAULT_FIT_VALUE + "--embedding" (no value) = 4
        ModelParameters p = new ModelParameters().enableEmbedding();
        String[] arr = p.toArray();
        assertThat(arr, arrayWithSize(4));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--embedding"));
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    @Test
    public void testToArrayMultipleParameters() {
        ModelParameters p = new ModelParameters().setThreads(4).enableEmbedding();
        String[] arr = p.toArray();
        // 1 (argv[0]) + 2 (--fit DEFAULT_FIT_VALUE) + 2 (--threads 4) + 1 (--embedding) = 6
        assertThat(arr, arrayWithSize(6));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--threads"));
        assertThat(list, hasItem("4"));
        assertThat(list, hasItem("--embedding"));
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    // -------------------------------------------------------------------------
    // Builder chaining returns same instance
    // -------------------------------------------------------------------------

    @Test
    public void testBuilderChainingReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertThat(p.setThreads(4), is(sameInstance(p)));
        assertThat(p.setGpuLayers(10), is(sameInstance(p)));
        assertThat(p.enableEmbedding(), is(sameInstance(p)));
    }

    // -------------------------------------------------------------------------
    // mmproj — vision model projection file/url
    // -------------------------------------------------------------------------

    @Test
    public void testSetMmproj() {
        ModelParameters p = new ModelParameters().setMmproj("/models/mmproj.gguf");
        assertThat(p.parameters.get("--mmproj"), is("/models/mmproj.gguf"));
    }

    @Test
    public void testSetMmprojUrl() {
        ModelParameters p = new ModelParameters().setMmprojUrl("https://example.com/mmproj.gguf");
        assertThat(p.parameters.get("--mmproj-url"), is("https://example.com/mmproj.gguf"));
    }

    @Test
    public void testEnableMmprojAuto() {
        ModelParameters p = new ModelParameters().enableMmprojAuto();
        assertThat(p.parameters, hasKey("--mmproj-auto"));
    }

    @Test
    public void testDisableMmprojAuto() {
        ModelParameters p = new ModelParameters().enableMmprojAuto().setMmprojAuto(false);
        assertThat(p.parameters, hasKey("--no-mmproj-auto"));
        assertThat(p.parameters, not(hasKey("--mmproj-auto")));
    }

    @Test
    public void testEnableMmprojOffload() {
        ModelParameters p = new ModelParameters().enableMmprojOffload();
        assertThat(p.parameters, hasKey("--mmproj-offload"));
    }

    @Test
    public void testDisableMmprojOffload() {
        ModelParameters p = new ModelParameters().enableMmprojOffload().setMmprojOffload(false);
        assertThat(p.parameters, hasKey("--no-mmproj-offload"));
        assertThat(p.parameters, not(hasKey("--mmproj-offload")));
    }

    // -------------------------------------------------------------------------
    // Reasoning format / budget — model-level defaults for thinking models
    // -------------------------------------------------------------------------

    @Test
    public void testSetReasoningFormatNone() {
        ModelParameters p = new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.NONE);
        assertThat(p.parameters.get("--reasoning-format"), is("none"));
    }

    @Test
    public void testSetReasoningFormatAuto() {
        ModelParameters p = new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.AUTO);
        assertThat(p.parameters.get("--reasoning-format"), is("auto"));
    }

    @Test
    public void testSetReasoningFormatDeepseek() {
        ModelParameters p = new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.DEEPSEEK);
        assertThat(p.parameters.get("--reasoning-format"), is("deepseek"));
    }

    @Test
    public void testSetReasoningFormatDeepseekLegacy() {
        ModelParameters p =
                new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.DEEPSEEK_LEGACY);
        assertThat(p.parameters.get("--reasoning-format"), is("deepseek-legacy"));
    }

    @Test
    public void testSetReasoningBudgetPositive() {
        ModelParameters p = new ModelParameters().setReasoningBudget(1024);
        assertThat(p.parameters.get("--reasoning-budget"), is("1024"));
    }

    @Test
    public void testSetReasoningBudgetDisabled() {
        ModelParameters p = new ModelParameters().setReasoningBudget(-1);
        assertThat(p.parameters.get("--reasoning-budget"), is("-1"));
    }

    // -------------------------------------------------------------------------
    // setSleepIdleSeconds
    // -------------------------------------------------------------------------

    @Test
    public void testSetSleepIdleSeconds() {
        ModelParameters p = new ModelParameters().setSleepIdleSeconds(60);
        assertThat(p.parameters.get("--sleep-idle-seconds"), is("60"));
    }

    @Test
    public void testSetSleepIdleSecondsZero() {
        ModelParameters p = new ModelParameters().setSleepIdleSeconds(0);
        assertThat(p.parameters.get("--sleep-idle-seconds"), is("0"));
    }

    // -------------------------------------------------------------------------
    // setClearIdle / setKvUnified — correct flag names (regression)
    // -------------------------------------------------------------------------

    @Test
    public void testSetClearIdleTrue_usesCacheIdleSlotsFlag() {
        ModelParameters p = new ModelParameters().setClearIdle(true);
        assertThat(p.parameters, hasKey("--cache-idle-slots"));
        assertThat(p.parameters, not(hasKey("--no-cache-idle-slots")));
    }

    @Test
    public void testSetClearIdleFalse_usesNoCacheIdleSlotsFlag() {
        ModelParameters p = new ModelParameters().setClearIdle(false);
        assertThat(p.parameters, hasKey("--no-cache-idle-slots"));
        assertThat(p.parameters, not(hasKey("--cache-idle-slots")));
    }

    // -------------------------------------------------------------------------
    // Plan C-list coverage gaps exposed (all within pinned b9941, no version bump)
    // -------------------------------------------------------------------------

    @Test
    public void testSetEmbdNormalize() {
        assertThat(new ModelParameters().setEmbdNormalize(1).parameters.get("--embd-normalize"), is("1"));
    }

    @Test
    public void testSetEmbdOutputFormat() {
        assertThat(new ModelParameters().setEmbdOutputFormat("f16").parameters.get("--embd-output-format"), is("f16"));
    }

    @Test
    public void testEnableCpuMoe_setsFlag() {
        assertThat(new ModelParameters().enableCpuMoe().parameters, hasKey("--cpu-moe"));
        assertThat(new ModelParameters().setCpuMoe(false).parameters, not(hasKey("--cpu-moe")));
        assertThat(new ModelParameters().setCpuMoe(true).parameters, hasKey("--cpu-moe"));
    }

    @Test
    public void testSetCpuMoeLayers() {
        assertThat(new ModelParameters().setCpuMoeLayers(8).parameters.get("--n-cpu-moe"), is("8"));
    }

    @Test
    public void testEnableCpuMoeDraft_setsFlag() {
        assertThat(new ModelParameters().enableCpuMoeDraft().parameters, hasKey("--cpu-moe-draft"));
        assertThat(new ModelParameters().setCpuMoeDraft(false).parameters, not(hasKey("--cpu-moe-draft")));
        assertThat(new ModelParameters().setCpuMoeDraft(true).parameters, hasKey("--cpu-moe-draft"));
    }

    @Test
    public void testEnableMtp_setsFlag() {
        assertThat(new ModelParameters().enableMtp().parameters, hasKey("--mtp"));
        assertThat(new ModelParameters().setMtp(false).parameters, not(hasKey("--mtp")));
        assertThat(new ModelParameters().setMtp(true).parameters, hasKey("--mtp"));
    }

    @Test
    public void testSetFitCtx() {
        assertThat(new ModelParameters().setFitCtx(4096).parameters.get("--fit-ctx"), is("4096"));
    }

    @Test
    public void testSetFitTarget() {
        assertThat(new ModelParameters().setFitTarget("MiB0,MiB1").parameters.get("--fit-target"), is("MiB0,MiB1"));
    }

    @Test
    public void testEnableFitPrint_setsFlag() {
        assertThat(new ModelParameters().enableFitPrint().parameters, hasKey("--fit-print"));
        assertThat(new ModelParameters().setFitPrint(false).parameters, not(hasKey("--fit-print")));
        assertThat(new ModelParameters().setFitPrint(true).parameters, hasKey("--fit-print"));
    }

    @Test
    public void testSetOverrideTensor() {
        assertThat(
                new ModelParameters()
                        .setOverrideTensor("tok_embeddings=Q4_0")
                        .parameters
                        .get("--override-tensor"),
                is("tok_embeddings=Q4_0"));
    }

    @Test
    public void testSetOverrideTensorDraft() {
        assertThat(
                new ModelParameters()
                        .setOverrideTensorDraft("blk.0=Q4_0")
                        .parameters
                        .get("--override-tensor-draft"),
                is("blk.0=Q4_0"));
    }

    @Test
    public void testSetPromptCache() {
        assertThat(
                new ModelParameters()
                        .setPromptCache("/tmp/cache.bin")
                        .parameters
                        .get("--prompt-cache"),
                is("/tmp/cache.bin"));
    }

    @Test
    public void testEnablePromptCacheRo_setsFlag() {
        assertThat(new ModelParameters().enablePromptCacheRo().parameters, hasKey("--prompt-cache-ro"));
        assertThat(new ModelParameters().setPromptCacheRo(false).parameters, not(hasKey("--prompt-cache-ro")));
        assertThat(new ModelParameters().setPromptCacheRo(true).parameters, hasKey("--prompt-cache-ro"));
    }

    @Test
    public void testEnablePromptCacheAll_setsFlag() {
        assertThat(new ModelParameters().enablePromptCacheAll().parameters, hasKey("--prompt-cache-all"));
        assertThat(new ModelParameters().setPromptCacheAll(false).parameters, not(hasKey("--prompt-cache-all")));
        assertThat(new ModelParameters().setPromptCacheAll(true).parameters, hasKey("--prompt-cache-all"));
    }

    @Test
    public void testSetImageMaxTokens() {
        assertThat(new ModelParameters().setImageMaxTokens(512).parameters.get("--image-max-tokens"), is("512"));
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setImageMaxTokens(-1));
    }

    @Test
    public void testSetImageMinTokens() {
        assertThat(new ModelParameters().setImageMinTokens(128).parameters.get("--image-min-tokens"), is("128"));
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setImageMinTokens(-1));
    }

    @Test
    public void testSetVideo() {
        assertThat(new ModelParameters().setVideo("/tmp/clip.mp4").parameters.get("--video"), is("/tmp/clip.mp4"));
    }

    @Test
    public void testSetMediaPath() {
        assertThat(new ModelParameters().setMediaPath("/tmp/media").parameters.get("--media-path"), is("/tmp/media"));
    }

    @Test
    public void testSetChunkSize() {
        assertThat(new ModelParameters().setChunkSize(256).parameters.get("--chunk-size"), is("256"));
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setChunkSize(-1));
    }

    @Test
    public void testSetChunkSeparator() {
        assertThat(new ModelParameters().setChunkSeparator("\n").parameters.get("--chunk-separator"), is("\n"));
    }

    @Test
    public void testSetChunks() {
        assertThat(new ModelParameters().setChunks(4).parameters.get("--chunks"), is("4"));
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setChunks(0));
    }

    @Test
    public void testSetClsSeparator() {
        assertThat(new ModelParameters().setClsSeparator("<cls>").parameters.get("--cls-separator"), is("<cls>"));
    }

    @Test
    public void testSetModelsDir() {
        assertThat(new ModelParameters().setModelsDir("/tmp/models").parameters.get("--models-dir"), is("/tmp/models"));
    }

    @Test
    public void testSetModelsMax() {
        assertThat(new ModelParameters().setModelsMax(4).parameters.get("--models-max"), is("4"));
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setModelsMax(-1));
    }

    @Test
    public void testEnableModelsAutoload_setsFlag() {
        assertThat(new ModelParameters().enableModelsAutoload().parameters, hasKey("--models-autoload"));
        assertThat(new ModelParameters().setModelsAutoload(false).parameters, not(hasKey("--models-autoload")));
        assertThat(new ModelParameters().setModelsAutoload(true).parameters, hasKey("--models-autoload"));
    }

    @Test
    public void testSetModelsPreset() {
        assertThat(
                new ModelParameters()
                        .setModelsPreset("/tmp/preset.json")
                        .parameters
                        .get("--models-preset"),
                is("/tmp/preset.json"));
    }
}
