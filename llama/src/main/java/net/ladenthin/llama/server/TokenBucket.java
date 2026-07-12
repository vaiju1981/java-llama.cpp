// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

/**
 * A single-key token-bucket rate limiter.
 *
 * <p>Tokens refill continuously at {@code refillTokensPerSecond}, up to {@code capacity}. Each
 * {@link #tryConsume()} takes one token; when the bucket is empty the call returns {@code false} so
 * the caller can reject the request. The class is internally synchronized, so one instance can be
 * shared across the server's request threads for a given bucket key (an API key or client address).
 */
final class TokenBucket {

    private final double capacity;
    private final double refillTokensPerSecond;
    private double tokens;
    private long lastRefillNanos;

    TokenBucket(double capacity, double refillTokensPerSecond) {
        this.capacity = Math.max(1.0, capacity);
        this.refillTokensPerSecond = refillTokensPerSecond;
        this.tokens = this.capacity;
        this.lastRefillNanos = System.nanoTime();
    }

    /**
     * Current token count, after refilling.
     *
     * @return the available tokens (for testing/inspection)
     */
    synchronized double available() {
        refill();
        return tokens;
    }

    /**
     * Try to consume one token.
     *
     * @return {@code true} if a token was available and consumed, {@code false} otherwise
     */
    synchronized boolean tryConsume() {
        refill();
        if (tokens >= 1.0) {
            tokens -= 1.0;
            return true;
        }
        return false;
    }

    private void refill() {
        long now = System.nanoTime();
        long elapsedNanos = now - lastRefillNanos;
        if (elapsedNanos <= 0) {
            return;
        }
        double added = (elapsedNanos / 1_000_000_000.0) * refillTokensPerSecond;
        if (added > 0.0) {
            tokens = Math.min(capacity, tokens + added);
            lastRefillNanos = now;
        }
    }
}
