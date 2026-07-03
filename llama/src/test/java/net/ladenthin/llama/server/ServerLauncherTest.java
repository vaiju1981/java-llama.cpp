// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayContaining;
import static org.hamcrest.Matchers.emptyArray;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * Pure-Java unit tests for {@link ServerLauncher}'s dispatch logic (selector detection + flag
 * stripping). No server is started and no native library is required.
 */
public class ServerLauncherTest {

    @Test
    public void selectsNativeByDefault() {
        assertThat(ServerLauncher.selectsOpenAiCompat(new String[] {"-m", "m.gguf", "--port", "8080"}), is(false));
    }

    @Test
    public void selectsOpenAiCompatWhenFlagPresent() {
        assertThat(ServerLauncher.selectsOpenAiCompat(new String[] {"--open-ai-compat", "-m", "m.gguf"}), is(true));
    }

    @Test
    public void selectorFlagPositionDoesNotMatter() {
        assertThat(ServerLauncher.selectsOpenAiCompat(new String[] {"-m", "m.gguf", "--open-ai-compat"}), is(true));
    }

    @Test
    public void withoutFlagStripsTheSelectorAndPreservesTheRest() {
        String[] out = ServerLauncher.withoutFlag(
                new String[] {"--open-ai-compat", "-m", "m.gguf", "--port", "8080"},
                ServerLauncher.OPEN_AI_COMPAT_FLAG);
        assertThat(out, arrayContaining("-m", "m.gguf", "--port", "8080"));
    }

    @Test
    public void withoutFlagRemovesEveryOccurrence() {
        String[] out = ServerLauncher.withoutFlag(
                new String[] {"--open-ai-compat", "-m", "m.gguf", "--open-ai-compat"},
                ServerLauncher.OPEN_AI_COMPAT_FLAG);
        assertThat(out, arrayContaining("-m", "m.gguf"));
    }

    @Test
    public void withoutFlagIsNoOpWhenAbsent() {
        String[] in = new String[] {"-m", "m.gguf"};
        assertThat(ServerLauncher.withoutFlag(in, ServerLauncher.OPEN_AI_COMPAT_FLAG), arrayContaining("-m", "m.gguf"));
    }

    @Test
    public void withoutFlagOnEmptyArgsIsEmpty() {
        assertThat(ServerLauncher.withoutFlag(new String[] {}, ServerLauncher.OPEN_AI_COMPAT_FLAG), is(emptyArray()));
    }
}
