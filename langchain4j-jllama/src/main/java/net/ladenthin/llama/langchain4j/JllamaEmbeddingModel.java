// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import net.ladenthin.llama.LlamaModel;

/**
 * langchain4j {@link EmbeddingModel} backed by an in-process java-llama.cpp model.
 *
 * <p>The backing {@link LlamaModel} must be loaded in embedding mode
 * ({@code ModelParameters.enableEmbedding()}). The model is <em>borrowed</em> (never closed here) —
 * see {@link JllamaChatModel}.
 */
public final class JllamaEmbeddingModel implements EmbeddingModel {

    private final LlamaModel model;

    /**
     * Creates an embedding model over a borrowed {@link LlamaModel}.
     *
     * @param model the loaded embedding-mode model to drive; not closed by this adapter
     */
    public JllamaEmbeddingModel(LlamaModel model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        List<Embedding> embeddings = new ArrayList<>(textSegments.size());
        for (TextSegment segment : textSegments) {
            embeddings.add(Embedding.from(model.embed(segment.text())));
        }
        return Response.from(embeddings);
    }
}
