// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * Cancellation handle for a blocking {@link LlamaModel} call. Pass an instance to
 * {@link LlamaModel#complete(InferenceParameters, CancellationToken)} and invoke
 * {@link #cancel()} from another thread to abort the inference loop.
 * <p>
 * Cancellation has two layers:
 * </p>
 * <ol>
 *   <li><b>Immediate, server-side</b>: while {@code complete(...)} is running it
 *       registers the token with the active native task id, so {@link #cancel()}
 *       can post a {@code SERVER_TASK_TYPE_CANCEL} message directly to the upstream
 *       server's task queue. The queue is mutex-locked internally, so this is safe
 *       from any thread. The native worker observes the cancel on its next slot
 *       iteration and releases the slot, which causes the in-flight token receive
 *       to return with a stop result. Effective latency is sub-token.</li>
 *   <li><b>Cooperative fallback</b>: {@link #cancel()} also flips a volatile flag
 *       that the inference loop polls between tokens. If the token is cancelled
 *       <em>before</em> {@code complete(...)} starts (or the model reference was
 *       lost), the flag still aborts the loop at the next token boundary.</li>
 * </ol>
 * <p>
 * The reader-backed buffer is intentionally <em>not</em> freed by
 * {@link #cancel()} &#x2014; that was the use-after-free root cause of the
 * previous mid-token attempt (a concurrent {@code rd-&gt;next()} held a raw
 * pointer into the erased {@code unique_ptr}). The native {@code queueCancel}
 * primitive posts the {@code SERVER_TASK_TYPE_CANCEL} task to the upstream
 * queue directly and does <em>not</em> touch the reader's
 * {@code waiting_task_ids} registration. That ordering is critical: removing
 * the registration would cause the worker's later {@code send()} of the slot's
 * stop result to be silently dropped, which would in turn leave the inference
 * thread's polling {@code recv_with_timeout} loop spinning forever (this was
 * observed as a CI hang after the first attempt at §2.10). The reader is
 * cleaned up by the normal stop-result code path in
 * {@code receiveCompletionJson} once the natural stop arrives.
 * </p>
 * <p>
 * A token may be reused across calls but must be used by only one inference at a
 * time. {@link #cancel()} and {@link #isCancelled()} are safe to invoke
 * concurrently with the inference loop.
 * </p>
 */
public final class CancellationToken {

    private volatile boolean cancelled;
    private volatile LlamaModel registeredModel;
    private volatile int registeredTaskId;

    /** Construct a fresh, not-cancelled token. */
    public CancellationToken() {
        // empty
    }

    /**
     * Cancellation flag accessor.
     * @return {@code true} once {@link #cancel()} has been called and before {@link #reset()}
     */
    public boolean isCancelled() {
        return cancelled;
    }

    /**
     * Request cancellation. Flips the cooperative flag and, if the token is
     * currently bound to a live native task, posts an immediate cancel to the
     * upstream server queue so the inference returns sub-token rather than at the
     * next token boundary. Idempotent and safe to call from any thread.
     */
    public void cancel() {
        cancelled = true;
        // Snapshot the registration; both fields are volatile, so this is a torn-
        // free read of the (model, taskId) pair as it stood when cancel() began.
        // If complete(...) has already unregistered, registeredModel is null and
        // we fall back to the cooperative path (which is a no-op because the
        // loop has already returned).
        LlamaModel m = registeredModel;
        int id = registeredTaskId;
        if (m != null) {
            try {
                m.queueCancel(id);
            } catch (RuntimeException ignored) {
                // queueCancel is best-effort; the cooperative flag still applies
                // and the loop will exit at the next token boundary even if the
                // native call failed (e.g. reader already removed naturally).
            }
        }
    }

    /** Clear the cancelled flag so the token can be reused. Package-private. */
    void reset() {
        cancelled = false;
    }

    /**
     * Bind this token to a live inference task. Called by
     * {@link LlamaModel#complete(InferenceParameters, CancellationToken)} just
     * after the task id is known and before the receive loop starts.
     */
    void register(LlamaModel model, int taskId) {
        this.registeredTaskId = taskId;
        this.registeredModel = model;
    }

    /**
     * Detach this token from its bound task. Called by
     * {@link LlamaModel#complete(InferenceParameters, CancellationToken)} in its
     * {@code finally} block so a subsequent {@link #cancel()} cannot reach into a
     * freed reader slot.
     */
    void unregister() {
        this.registeredModel = null;
    }
}
