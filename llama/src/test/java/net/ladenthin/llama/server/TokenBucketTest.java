// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Model-free unit tests for the {@link TokenBucket} rate-limiter primitive. */
public class TokenBucketTest {

    @Test
    public void startsFull() {
        TokenBucket bucket = new TokenBucket(5.0, 0.0);
        assertEquals(5.0, bucket.available(), 1e-9);
    }

    @Test
    public void consumesUpToCapacityThenDenies() {
        // Zero refill rate makes the outcome deterministic: capacity tokens, then nothing.
        TokenBucket bucket = new TokenBucket(2.0, 0.0);
        assertTrue(bucket.tryConsume());
        assertTrue(bucket.tryConsume());
        assertFalse(bucket.tryConsume());
        assertEquals(0.0, bucket.available(), 1e-9);
    }

    @Test
    public void capacityIsAtLeastOne() {
        TokenBucket bucket = new TokenBucket(0.5, 10.0);
        assertEquals(1.0, bucket.available(), 1e-9);
    }
}
