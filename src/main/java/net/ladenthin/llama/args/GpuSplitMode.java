package net.ladenthin.llama.args;

/**
 * GPU tensor split mode for {@code --split-mode}.
 */
public enum GpuSplitMode implements CliArg {

    /** No split; use a single GPU. */
    NONE("none"),
    /** Split by transformer layer across GPUs. */
    LAYER("layer"),
    /** Split by tensor row across GPUs. */
    ROW("row");

    private final String argValue;

    GpuSplitMode(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
