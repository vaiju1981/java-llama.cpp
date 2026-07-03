// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.ArrayList;
import java.util.List;

/**
 * Fat-jar entry point that dispatches to one of the two server modes based on a single selector
 * flag. With {@value #OPEN_AI_COMPAT_FLAG} present it runs {@link OpenAiCompatServer} (the
 * Java-transport, OpenAI-compatible JSON API); without it, {@link NativeServer} (the full native
 * llama.cpp server with embedded WebUI, the default).
 *
 * <p>Every other argument is forwarded verbatim to the chosen server; the {@value
 * #OPEN_AI_COMPAT_FLAG} marker itself is stripped so it never reaches either parser (it is not a
 * llama.cpp flag, and {@code llama_server} rejects unknown flags).</p>
 *
 * <p><strong>Flag sets differ.</strong> {@link NativeServer} forwards <em>every</em> llama-server
 * flag to {@code llama_server}, whereas {@link OpenAiCompatServer}'s CLI ({@link OpenAiServerCli})
 * accepts a curated subset and rejects unknown flags — so native-only flags (e.g. {@code --ui},
 * {@code -fa}) cannot be combined with {@value #OPEN_AI_COMPAT_FLAG}.</p>
 *
 * <p>Both underlying mains remain directly runnable by class name via {@code java -cp}; this
 * launcher is purely a convenience so a single {@code java -jar} covers both.</p>
 */
public final class ServerLauncher {

    /** Selector flag: when present, run {@link OpenAiCompatServer} instead of the default {@link NativeServer}. */
    public static final String OPEN_AI_COMPAT_FLAG = "--open-ai-compat";

    private ServerLauncher() {}

    /**
     * Dispatches to {@link OpenAiCompatServer#main(String[])} when {@value #OPEN_AI_COMPAT_FLAG} is
     * present (with that marker removed from the arguments), otherwise to
     * {@link NativeServer#main(String[])} with all arguments forwarded unchanged.
     *
     * @param args the process arguments
     * @throws Exception if the selected server's {@code main} throws (it blocks until shutdown)
     */
    public static void main(String[] args) throws Exception {
        if (selectsOpenAiCompat(args)) {
            OpenAiCompatServer.main(withoutFlag(args, OPEN_AI_COMPAT_FLAG));
        } else {
            NativeServer.main(args);
        }
    }

    /**
     * Whether the arguments request the OpenAI-compatible server via {@value #OPEN_AI_COMPAT_FLAG}.
     *
     * @param args the process arguments
     * @return {@code true} if the selector flag is present
     */
    static boolean selectsOpenAiCompat(String[] args) {
        for (final String arg : args) {
            if (OPEN_AI_COMPAT_FLAG.equals(arg)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns a copy of {@code args} with every occurrence of {@code flag} removed, preserving the
     * order of the remaining arguments.
     *
     * @param args the arguments
     * @param flag the flag token to strip
     * @return a new array without {@code flag}
     */
    static String[] withoutFlag(String[] args, String flag) {
        final List<String> filtered = new ArrayList<>(args.length);
        for (final String arg : args) {
            if (!flag.equals(arg)) {
                filtered.add(arg);
            }
        }
        return filtered.toArray(new String[0]);
    }
}
