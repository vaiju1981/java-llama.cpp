// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

/**
 * Thin multi-turn conversation wrapper over a {@link LlamaModel} slot. Maintains an
 * accumulating list of {@link ChatMessage} turns and forwards each {@link #send(String)}
 * to the underlying chat-completion API with the full transcript so far. KV-cache state
 * for the bound slot can be persisted via {@link #save(String)} and restored with
 * {@link #restore(String)}, which delegate to {@link LlamaModel#saveSlot(int, String)}
 * and {@link LlamaModel#restoreSlot(int, String)}.
 * <p>
 * Thread-safety: all public methods are serialized on a private intrinsic lock, so
 * concurrent {@link #send(String)} calls from multiple threads produce a well-formed
 * transcript with strict user/assistant alternation. {@link #stream(String)} sets a
 * "streaming in progress" flag and returns the iterator without holding the lock;
 * while that flag is set, {@link #send(String)}, a second {@link #stream(String)},
 * {@link #save(String)}, and {@link #restore(String)} fail-fast with
 * {@link IllegalStateException} until the caller invokes
 * {@link #commitStreamedReply(String)}.
 * </p>
 */
public final class Session implements AutoCloseable {

    private final LlamaModel model;
    private final int slotId;
    private final String systemMessage;
    private final List<Pair<String, String>> turns = new ArrayList<Pair<String, String>>();
    private final Consumer<InferenceParameters> paramsCustomizer;
    private final Object lock = new Object();
    private boolean streamingActive;

    /**
     * Create a session bound to a specific slot id, with an optional system prompt
     * applied to every {@link #send(String)} call.
     *
     * @param model the underlying model
     * @param slotId the slot id used by {@link #save(String)} / {@link #restore(String)}
     * @param systemMessage optional system prompt (may be {@code null} or empty)
     */
    public Session(LlamaModel model, int slotId, String systemMessage) {
        this(model, slotId, systemMessage, null);
    }

    /**
     * Create a session with a customizer that gets to mutate the
     * {@link InferenceParameters} for every call (e.g. set temperature, n_predict).
     *
     * @param model the underlying model
     * @param slotId the slot id
     * @param systemMessage optional system prompt
     * @param paramsCustomizer applied to each request's parameters; may be {@code null}
     */
    public Session(LlamaModel model, int slotId, String systemMessage, Consumer<InferenceParameters> paramsCustomizer) {
        this.model = model;
        this.slotId = slotId;
        this.systemMessage = systemMessage;
        this.paramsCustomizer = paramsCustomizer;
    }

    /**
     * Send a user message and return the assistant's text reply, appending both to the transcript.
     *
     * @param userMessage the user turn to append before invoking the model
     * @return the assistant's reply text
     */
    public String send(String userMessage) {
        synchronized (lock) {
            if (streamingActive) {
                throw new IllegalStateException("stream in progress; call commitStreamedReply(...) before send(...)");
            }
            turns.add(new Pair<String, String>("user", userMessage));
            InferenceParameters params = buildParams();
            try {
                String reply = model.chatCompleteText(params);
                turns.add(new Pair<String, String>("assistant", reply));
                return reply;
            } catch (RuntimeException e) {
                turns.remove(turns.size() - 1);
                throw e;
            }
        }
    }

    /**
     * Streaming variant of {@link #send(String)}. The returned iterable yields chunks of
     * the assistant reply; consume it fully (or via try-with-resources) before calling
     * {@link #send(String)} again, because the assistant turn is only appended to the
     * transcript when the caller invokes {@link #commitStreamedReply(String)}.
     *
     * @param userMessage the user turn to append before starting the stream
     * @return a {@link LlamaIterable} that yields assistant reply chunks
     */
    public LlamaIterable stream(String userMessage) {
        synchronized (lock) {
            if (streamingActive) {
                throw new IllegalStateException("stream in progress; call commitStreamedReply(...) before stream(...)");
            }
            turns.add(new Pair<String, String>("user", userMessage));
            try {
                LlamaIterable iterable = model.generateChat(buildParams());
                streamingActive = true;
                return iterable;
            } catch (RuntimeException e) {
                turns.remove(turns.size() - 1);
                throw e;
            }
        }
    }

    /**
     * Record an assistant reply that was produced by a previous {@link #stream(String)}
     * call. Called by the caller after it has accumulated the streamed text.
     *
     * @param assistantText the assistant text accumulated from a prior {@link #stream(String)} call
     */
    public void commitStreamedReply(String assistantText) {
        synchronized (lock) {
            if (!streamingActive) {
                throw new IllegalStateException("no stream in progress; call stream(...) first");
            }
            turns.add(new Pair<String, String>("assistant", assistantText));
            streamingActive = false;
        }
    }

    /**
     * Save this session's slot KV cache to {@code filepath}.
     *
     * @param filepath destination file path passed to {@link LlamaModel#saveSlot(int, String)}
     * @return the JSON response from the native save action
     */
    public String save(String filepath) {
        synchronized (lock) {
            if (streamingActive) {
                throw new IllegalStateException("stream in progress; call commitStreamedReply(...) before save(...)");
            }
            return model.saveSlot(slotId, filepath);
        }
    }

    /**
     * Restore this session's slot KV cache from {@code filepath}.
     *
     * @param filepath source file path passed to {@link LlamaModel#restoreSlot(int, String)}
     * @return the JSON response from the native restore action
     */
    public String restore(String filepath) {
        synchronized (lock) {
            if (streamingActive) {
                throw new IllegalStateException(
                        "stream in progress; call commitStreamedReply(...) before restore(...)");
            }
            return model.restoreSlot(slotId, filepath);
        }
    }

    /**
     * Transcript accessor.
     * @return the accumulated transcript so far, in order, including the system message if any
     */
    public List<ChatMessage> getMessages() {
        synchronized (lock) {
            List<ChatMessage> out = new ArrayList<ChatMessage>(turns.size() + 1);
            if (systemMessage != null && !systemMessage.isEmpty()) {
                out.add(new ChatMessage("system", systemMessage));
            }
            for (Pair<String, String> p : turns) {
                out.add(new ChatMessage(p.getKey(), p.getValue()));
            }
            return Collections.unmodifiableList(out);
        }
    }

    /** Erase the bound slot's KV cache. Does not modify the in-memory transcript. */
    @Override
    public void close() {
        synchronized (lock) {
            model.eraseSlot(slotId);
        }
    }

    private InferenceParameters buildParams() {
        InferenceParameters params =
                new InferenceParameters("").setMessages(systemMessage, new ArrayList<Pair<String, String>>(turns));
        if (paramsCustomizer != null) {
            paramsCustomizer.accept(params);
        }
        return params;
    }
}
