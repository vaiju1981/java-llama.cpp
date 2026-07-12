// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import org.jspecify.annotations.Nullable;

/**
 * Per-client guardrails for {@link OpenAiCompatServer}: a token-bucket rate limit (keyed by the
 * presented bearer token, or by client address when authentication is off) plus an optional
 * concurrency gate that caps the number of requests being served at once.
 *
 * <p>Upstream llama.cpp has limited native rate limiting, so this is the realistic Java-side path
 * called for by the serving-server plan. When both limits are disabled ({@code rateLimitRps <= 0}
 * and {@code maxConcurrentClients <= 0}) the filter is a no-op and {@link #isEnabled()} returns
 * {@code false} so the server does not even attach it.
 */
final class RateLimitFilter extends Filter {

    private static final int HTTP_TOO_MANY_REQUESTS = 429;

    private final double rateLimitRps;
    private final @Nullable String configuredKey;
    private final Map<String, TokenBucket> buckets = new ConcurrentHashMap<>();
    private final @Nullable Semaphore concurrency;

    RateLimitFilter(double rateLimitRps, int maxConcurrentClients, @Nullable String configuredKey) {
        this.rateLimitRps = rateLimitRps;
        this.configuredKey = configuredKey;
        this.concurrency = maxConcurrentClients > 0 ? new Semaphore(maxConcurrentClients) : null;
    }

    /**
     * Whether either guardrail is active.
     *
     * @return {@code true} if the rate limit or the concurrency gate is enabled
     */
    boolean isEnabled() {
        return rateLimitRps > 0.0 || concurrency != null;
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        // Let the CORS filter own OPTIONS preflights; never rate-limit them.
        if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
            chain.doFilter(exchange);
            return;
        }
        boolean acquired = true;
        if (concurrency != null) {
            acquired = concurrency.tryAcquire();
        }
        if (!acquired) {
            sendTooMany(exchange);
            return;
        }
        boolean allowed = true;
        if (rateLimitRps > 0.0) {
            String key = bucketKey(exchange);
            TokenBucket bucket = buckets.computeIfAbsent(key, k -> new TokenBucket(rateLimitRps, rateLimitRps));
            allowed = bucket.tryConsume();
        }
        if (!allowed) {
            if (concurrency != null) {
                concurrency.release();
            }
            sendTooMany(exchange);
            return;
        }
        try {
            chain.doFilter(exchange);
        } finally {
            if (concurrency != null) {
                concurrency.release();
            }
        }
    }

    private String bucketKey(HttpExchange exchange) {
        if (configuredKey != null && !configuredKey.isEmpty()) {
            String header = exchange.getRequestHeaders().getFirst("Authorization");
            if (header != null && header.startsWith("Bearer ")) {
                return header.substring(7);
            }
            return "anonymous";
        }
        InetSocketAddress addr = exchange.getRemoteAddress();
        return addr != null ? addr.getAddress().getHostAddress() : "unknown";
    }

    private static void sendTooMany(HttpExchange exchange) throws IOException {
        byte[] body = "{\"error\":\"rate limit exceeded\"}".getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(HTTP_TOO_MANY_REQUESTS, body.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(body);
        }
    }

    @Override
    public String description() {
        return "token-bucket rate limit + concurrency gate";
    }
}
