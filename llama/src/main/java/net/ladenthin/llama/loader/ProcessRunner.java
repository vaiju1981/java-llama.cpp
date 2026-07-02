// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import lombok.ToString;

@ToString
class ProcessRunner {

    private final Java8CompatibilityHelper compatibilityHelper = new Java8CompatibilityHelper();

    String runAndWaitFor(String command) throws IOException, InterruptedException {
        Process p = Runtime.getRuntime().exec(splitArgs(command));
        p.waitFor();

        return getProcessOutput(p);
    }

    String runAndWaitFor(String command, long timeout, TimeUnit unit) throws IOException, InterruptedException {
        Process p = Runtime.getRuntime().exec(splitArgs(command));
        p.waitFor(timeout, unit);

        return getProcessOutput(p);
    }

    /**
     * Split a space-delimited command string into an argv array so that
     * {@link Runtime#exec(String[])} (rather than the shell-tokenising
     * {@link Runtime#exec(String)}) can be used. This avoids command-injection
     * concerns from the latter — callers only pass simple whitespace-separated
     * commands such as {@code "uname -o"}.
     */
    private static String[] splitArgs(String command) {
        return command.split(" ");
    }

    private String getProcessOutput(Process process) throws IOException {
        try (InputStream in = process.getInputStream()) {
            int readLen;
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            byte[] buf = new byte[32];
            while ((readLen = in.read(buf, 0, buf.length)) >= 0) {
                b.write(buf, 0, readLen);
            }
            return compatibilityHelper.toString(b, StandardCharsets.UTF_8);
        }
    }
}
