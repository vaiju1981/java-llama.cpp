// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Model-free smoke test for the {@link NativeServer} scaffold: it must construct without any native
 * work, expose its configured host/port, never report itself running, throw a clear
 * {@link UnsupportedOperationException} from {@link NativeServer#start()} until the native routes are
 * wired, and be a safe no-op {@link AutoCloseable}. No model and no {@code libjllama} required.
 */
public class NativeServerSmokeTest {

    private static OpenAiServerConfig config() {
        return OpenAiServerConfig.builder().host("127.0.0.1").port(1234).build();
    }

    @Test
    public void exposesConfiguredHostAndPortWithoutStarting() {
        NativeServer server = new NativeServer(config());
        assertThat(server.getHost(), is("127.0.0.1"));
        assertThat(server.getPort(), is(1234));
        assertThat(server.isRunning(), is(false));
    }

    @Test
    public void startThrowsUntilNativeRoutesAreWired() {
        NativeServer server = new NativeServer(config());
        UnsupportedOperationException ex = assertThrows(UnsupportedOperationException.class, server::start);
        assertThat(ex.getMessage(), containsString("not yet wired"));
        assertThat(server.isRunning(), is(false));
    }

    @Test
    public void closeIsSafeNoOpEvenViaTryWithResources() {
        try (NativeServer server = new NativeServer(config())) {
            assertThat(server.isRunning(), is(false));
        }
    }
}
