// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;

/**
 * Stateful translator that turns the OpenAI streaming chat chunks into the OpenAI Responses SSE event
 * sequence: {@code response.created} → (for a text message) {@code response.output_item.added} +
 * {@code response.content_part.added} + {@code response.output_text.delta}* +
 * {@code response.output_text.done} + {@code response.content_part.done} +
 * {@code response.output_item.done} → (per tool call) a {@code function_call} item with
 * {@code response.function_call_arguments.done} → {@code response.completed}. Each event carries a
 * monotonic {@code sequence_number}.
 *
 * <p>Text deltas are emitted live; tool calls are reconstructed via {@link ToolCallDeltaAccumulator} and
 * emitted as whole {@code function_call} items at the end. Free of JNI / model dependencies;
 * unit-testable by feeding chunk JSON.
 */
final class ResponsesStreamTranslator {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final String model;
    private final String responseId;
    private final String messageItemId;
    private final ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
    private final StringBuilder text = new StringBuilder();

    private int sequence;
    private boolean messageOpen;
    private int nextOutputIndex;
    private int messageOutputIndex = -1;

    ResponsesStreamTranslator(String model, String responseId) {
        this.model = model;
        this.responseId = responseId;
        this.messageItemId = "msg_" + responseId;
    }

    /**
     * The opening {@code response.created} event.
     *
     * @return the framed SSE event
     */
    String begin() {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.set("response", ResponsesApiSupport.newResponseShell(model, responseId, "in_progress"));
        return ResponsesApiSupport.sseEvent("response.created", sequence++, data);
    }

    /**
     * Translate one OpenAI chunk into the Responses events it produces (opening the message item and
     * content part on first text, then text deltas), accumulating tool-call fragments. Returns an empty
     * string when the chunk yields no event.
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @return zero or more framed SSE events, concatenated
     */
    String onChunk(String openAiChunkJson) {
        StringBuilder out = new StringBuilder();
        try {
            JsonNode chunk = OBJECT_MAPPER.readTree(openAiChunkJson);
            accumulator.accept(chunk);
            JsonNode content = chunk.path("choices").path(0).path("delta").path("content");
            if (content.isTextual() && !content.asText().isEmpty()) {
                if (!messageOpen) {
                    messageOutputIndex = nextOutputIndex++;
                    out.append(outputItemAdded(messageOutputIndex, messageItemShell()));
                    out.append(contentPartAdded());
                    messageOpen = true;
                }
                String delta = content.asText();
                text.append(delta);
                ObjectNode data = ResponsesApiSupport.dataObject();
                data.put("item_id", messageItemId);
                data.put("output_index", messageOutputIndex);
                data.put("content_index", 0);
                data.put("delta", delta);
                out.append(ResponsesApiSupport.sseEvent("response.output_text.delta", sequence++, data));
            }
        } catch (IOException e) {
            // A malformed chunk produces no events.
        }
        return out.toString();
    }

    /**
     * The closing events: finish the text content part / message item, emit a {@code function_call} item
     * per accumulated tool call, then {@code response.completed} carrying the assembled output and usage.
     *
     * @return the framed SSE events, concatenated
     */
    String end() {
        StringBuilder out = new StringBuilder();
        ArrayNode output = OBJECT_MAPPER.createArrayNode();

        if (messageOpen) {
            ObjectNode textDone = ResponsesApiSupport.dataObject();
            textDone.put("item_id", messageItemId);
            textDone.put("output_index", messageOutputIndex);
            textDone.put("content_index", 0);
            textDone.put("text", text.toString());
            out.append(ResponsesApiSupport.sseEvent("response.output_text.done", sequence++, textDone));

            ObjectNode partDone = ResponsesApiSupport.dataObject();
            partDone.put("item_id", messageItemId);
            partDone.put("output_index", messageOutputIndex);
            partDone.put("content_index", 0);
            ObjectNode part = partDone.putObject("part");
            part.put("type", "output_text");
            part.put("text", text.toString());
            out.append(ResponsesApiSupport.sseEvent("response.content_part.done", sequence++, partDone));

            ObjectNode messageItem = completedMessageItem();
            output.add(messageItem);
            out.append(outputItemDone(messageOutputIndex, messageItem));
        }

        for (JsonNode toolCall : accumulator.toOpenAiToolCalls()) {
            int index = nextOutputIndex++;
            ObjectNode functionCall = ResponsesApiSupport.functionCallItem(toolCall);
            out.append(outputItemAdded(index, functionCall));
            ObjectNode argsDone = ResponsesApiSupport.dataObject();
            argsDone.put("item_id", functionCall.path("id").asText());
            argsDone.put("output_index", index);
            argsDone.put(
                    "arguments", toolCall.path("function").path("arguments").asText(""));
            out.append(ResponsesApiSupport.sseEvent("response.function_call_arguments.done", sequence++, argsDone));
            out.append(outputItemDone(index, functionCall));
            output.add(functionCall);
        }

        ObjectNode completed = ResponsesApiSupport.dataObject();
        ObjectNode response = ResponsesApiSupport.newResponseShell(model, responseId, "completed");
        response.set("output", output);
        completed.set("response", response);
        out.append(ResponsesApiSupport.sseEvent("response.completed", sequence++, completed));
        return out.toString();
    }

    private ObjectNode messageItemShell() {
        ObjectNode item = OBJECT_MAPPER.createObjectNode();
        item.put("type", "message");
        item.put("id", messageItemId);
        item.put("status", "in_progress");
        item.put("role", "assistant");
        item.putArray("content");
        return item;
    }

    private ObjectNode completedMessageItem() {
        ObjectNode item = OBJECT_MAPPER.createObjectNode();
        item.put("type", "message");
        item.put("id", messageItemId);
        item.put("status", "completed");
        item.put("role", "assistant");
        ObjectNode textPart = item.putArray("content").addObject();
        textPart.put("type", "output_text");
        textPart.put("text", text.toString());
        textPart.putArray("annotations");
        return item;
    }

    private String outputItemAdded(int outputIndex, ObjectNode item) {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("output_index", outputIndex);
        data.set("item", item);
        return ResponsesApiSupport.sseEvent("response.output_item.added", sequence++, data);
    }

    private String outputItemDone(int outputIndex, ObjectNode item) {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("output_index", outputIndex);
        data.set("item", item);
        return ResponsesApiSupport.sseEvent("response.output_item.done", sequence++, data);
    }

    private String contentPartAdded() {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("item_id", messageItemId);
        data.put("output_index", messageOutputIndex);
        data.put("content_index", 0);
        ObjectNode part = data.putObject("part");
        part.put("type", "output_text");
        part.put("text", "");
        return ResponsesApiSupport.sseEvent("response.content_part.added", sequence++, data);
    }
}
