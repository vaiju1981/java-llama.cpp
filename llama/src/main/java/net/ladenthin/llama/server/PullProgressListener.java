// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import org.jspecify.annotations.Nullable;

/**
 * Reports download progress. Implementations must be cheap to call repeatedly.
 */
@FunctionalInterface
public interface PullProgressListener {

    /**
     * Called as bytes arrive during a download.
     *
     * @param bytesDownloaded bytes received so far
     * @param totalBytes expected total, or {@code null} when the server did not advertise a size
     */
    void onProgress(long bytesDownloaded, @Nullable Long totalBytes);
}
