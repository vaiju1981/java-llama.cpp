// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

/**
 * Result of a single-model health probe performed by {@link ModelPool#getModelHealth(String)}.
 */
public final class ModelHealth {

    private final String alias;
    private final boolean healthy;
    private final long latencyMillis;
    private final String raw;

    /**
     * @param alias the model alias the probe was requested for
     * @param healthy whether the server reported a healthy status
     * @param latencyMillis round-trip latency to the server's {@code /health} endpoint, in milliseconds
     * @param raw the raw {@code /health} response body
     */
    public ModelHealth(String alias, boolean healthy, long latencyMillis, String raw) {
        this.alias = alias;
        this.healthy = healthy;
        this.latencyMillis = latencyMillis;
        this.raw = raw;
    }

    /**
     * The model alias the probe was requested for.
     *
     * @return the model alias
     */
    public String getAlias() {
        return alias;
    }

    /**
     * Whether the server reported a healthy status.
     *
     * @return {@code true} if healthy
     */
    public boolean isHealthy() {
        return healthy;
    }

    /**
     * Round-trip latency to the server's {@code /health} endpoint.
     *
     * @return latency in milliseconds
     */
    public long getLatencyMillis() {
        return latencyMillis;
    }

    /**
     * The raw {@code /health} response body.
     *
     * @return the raw response text
     */
    public String getRaw() {
        return raw;
    }
}
