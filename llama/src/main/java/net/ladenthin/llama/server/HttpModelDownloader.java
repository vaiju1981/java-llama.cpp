// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Pure-Java {@link ModelDownloader} using {@link HttpURLConnection}. Follows redirects and honours
 * {@link ResolvedModelSource#getHeaders()} (e.g. a HuggingFace token). The file is streamed to a
 * temporary name and atomically moved into place on success.
 *
 * <p>Model-free: exercises cleanly against a stub {@code com.sun.net.httpserver.HttpServer}.
 */
public final class HttpModelDownloader implements ModelDownloader {

    private static final int CONNECT_TIMEOUT_MS = 30_000;
    private static final int READ_TIMEOUT_MS = 30_000;

    @Override
    public Path download(ResolvedModelSource source, Path targetDir, PullProgressListener listener) throws IOException {
        String url = source.getUrl();
        if (url == null) {
            throw new IllegalArgumentException("HttpModelDownloader requires a remote source");
        }
        Files.createDirectories(targetDir);
        String fileName = fileNameFor(url);
        Path temp = targetDir.resolve(fileName + ".part");
        Path finalPath = targetDir.resolve(fileName);

        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setInstanceFollowRedirects(true);
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);
        conn.setRequestProperty("User-Agent", "java-llama.cpp-model-puller");
        for (java.util.Map.Entry<String, String> h : source.getHeaders().entrySet()) {
            conn.setRequestProperty(h.getKey(), h.getValue());
        }
        try {
            int code = conn.getResponseCode();
            if (code < 200 || code >= 300) {
                throw new IOException("Download failed: HTTP " + code + " for " + url);
            }
            long total = conn.getContentLengthLong();
            Long totalBoxed = total >= 0 ? Long.valueOf(total) : null;
            try (InputStream in = conn.getInputStream();
                    OutputStream out = Files.newOutputStream(temp)) {
                byte[] buf = new byte[64 * 1024];
                long received = 0;
                int n;
                while ((n = in.read(buf)) != -1) {
                    out.write(buf, 0, n);
                    received += n;
                    if (listener != null) {
                        listener.onProgress(received, totalBoxed);
                    }
                }
            }
            Files.move(temp, finalPath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            return finalPath;
        } finally {
            conn.disconnect();
            try {
                Files.deleteIfExists(temp);
            } catch (IOException ignored) {
                // best-effort cleanup of a partial download
            }
        }
    }

    private static String fileNameFor(String url) {
        int slash = Math.max(url.lastIndexOf('/'), url.lastIndexOf('\\'));
        String name = slash >= 0 && slash + 1 < url.length() ? url.substring(slash + 1) : url;
        int q = name.indexOf('?');
        if (q >= 0) {
            name = name.substring(0, q);
        }
        if (!name.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf")) {
            name = name + ".gguf";
        }
        return name;
    }
}
