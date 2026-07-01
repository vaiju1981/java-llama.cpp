// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import net.ladenthin.llama.LlamaModel;

/**
 * langchain4j {@link ScoringModel} (re-ranker) backed by an in-process java-llama.cpp model.
 *
 * <p>Maps onto java-llama.cpp's native rerank endpoint, so the backing {@link LlamaModel} must be
 * loaded in reranking mode ({@code ModelParameters.enableReranking()}). Scores are returned in the
 * same order as the input segments. The model is <em>borrowed</em> (never closed here) — see
 * {@link JllamaChatModel}.
 */
public final class JllamaScoringModel implements ScoringModel {

    private final LlamaModel model;

    /**
     * Creates a scoring model over a borrowed {@link LlamaModel}.
     *
     * @param model the loaded reranking-mode model to drive; not closed by this adapter
     */
    public JllamaScoringModel(LlamaModel model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        String[] documents = new String[segments.size()];
        for (int i = 0; i < segments.size(); i++) {
            documents[i] = segments.get(i).text();
        }
        double[] scores = LangChain4jMapping.parseRerankScores(model.handleRerank(query, documents), documents.length);
        List<Double> result = new ArrayList<>(scores.length);
        for (double score : scores) {
            result.add(score);
        }
        return Response.from(result);
    }
}
