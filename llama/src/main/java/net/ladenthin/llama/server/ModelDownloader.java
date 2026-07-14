// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Downloads a {@link ResolvedModelSource} to a file under {@code targetDir}. Transport-agnostic so
 * the puller can be unit-tested against a stub server and, in production, against real endpoints.
 */
public interface ModelDownloader {

    /**
     * Download the source into {@code targetDir}.
     *
     * @param source the resolved source (must be {@link ResolvedModelSource#isRemote() remote})
     * @param targetDir the directory to write the model file into
     * @param listener optional progress callback (may be {@code null})
     * @return the path of the downloaded file
     * @throws IOException if the download fails
     */
    Path download(ResolvedModelSource source, Path targetDir, PullProgressListener listener) throws IOException;
}
