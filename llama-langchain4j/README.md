# llama-langchain4j

[LangChain4j](https://github.com/langchain4j/langchain4j) adapters backed by an **in-process**
[java-llama.cpp](https://github.com/bernardladenthin/java-llama.cpp) model over JNI — no HTTP server,
no separate process.

This is a **separate Maven artifact** on purpose: it depends on `langchain4j-core`, but the core
`net.ladenthin:llama` binding does **not** depend on langchain4j, so plain java-llama.cpp users never
pull langchain4j (or its Java 17 floor) transitively.

> **Already have an OpenAI-compatible setup?** java-llama.cpp also ships
> `net.ladenthin.llama.server.OpenAiCompatServer`, so you can point langchain4j's `langchain4j-open-ai`
> client at a running server with zero code from this module. Use *this* module when you want the
> in-process path (no HTTP hop, single process — e.g. desktop/Android/embedded).

## Adapters

| Class | langchain4j interface | java-llama.cpp call |
|-------|-----------------------|---------------------|
| `JllamaChatModel` | `ChatModel` | `LlamaModel.chat(...)` |
| `JllamaStreamingChatModel` | `StreamingChatModel` | `LlamaModel.generateChat(...)` (token streaming) |
| `JllamaEmbeddingModel` | `EmbeddingModel` | `LlamaModel.embed(...)` |
| `JllamaScoringModel` | `ScoringModel` (re-ranking) | `LlamaModel.handleRerank(...)` |

## Lifecycle: the model is *borrowed*

Every adapter takes a `LlamaModel` you already loaded and **keeps owning**. The adapter never loads
or closes the native model — you manage it (try-with-resources or explicit `close()`). One
`LlamaModel` can back several adapters at once.

```java
try (LlamaModel llama = new LlamaModel(new ModelParameters().setModel("models/qwen3-0.6b.gguf"))) {
    ChatModel chat = new JllamaChatModel(llama);

    String reply = chat.chat("Write a haiku about lazy senior devs.");
    System.out.println(reply);
}
```

Streaming:

```java
StreamingChatModel chat = new JllamaStreamingChatModel(llama);
chat.chat("Tell me a story.", new StreamingChatResponseHandler() {
    @Override public void onPartialResponse(String token) { System.out.print(token); }
    @Override public void onCompleteResponse(ChatResponse response) { /* done */ }
    @Override public void onError(Throwable error) { error.printStackTrace(); }
});
```

Embeddings (model loaded with `enableEmbedding()`) and re-ranking
(`enableReranking()`) plug straight into langchain4j RAG:

```java
EmbeddingModel embeddings = new JllamaEmbeddingModel(embeddingLlama);
ScoringModel reranker     = new JllamaScoringModel(rerankLlama);
```

## Dependency

```xml
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama-langchain4j</artifactId>
    <version>5.0.5</version>
</dependency>
```

`langchain4j-core` is pulled transitively. You still supply a java-llama.cpp native library for your
platform the usual way (bundled in the `net.ladenthin:llama` JAR or on `java.library.path`).

## Building

This is a **reactor module** built, versioned and released together with the core (see the root
`pom.xml` `<modules>`). To build/test just this module locally, build it with its ancestors from the
repo root so the core is compiled/installed first:

```bash
# from the repo root: build the core (-am) and this module together
mvn -pl llama-langchain4j -am -DskipTests test

# or install the core first, then build here on its own
mvn -pl llama -am -DskipTests install
cd llama-langchain4j && mvn test
```

The model-backed integration tests self-skip unless you point them at a GGUF. Each adapter has
its own property so you can run them independently (a chat/instruct model, an embedding-mode model,
and a reranking-mode model respectively):

```bash
# chat + streaming (JllamaChatModelIntegrationTest)
mvn test -Dnet.ladenthin.llama.model.path=/abs/path/to/chat.gguf
# embeddings (JllamaEmbeddingModelIntegrationTest)
mvn test -Dnet.ladenthin.llama.langchain4j.embedding.model=/abs/path/to/embedding.gguf
# re-ranking / scoring (JllamaScoringModelIntegrationTest)
mvn test -Dnet.ladenthin.llama.langchain4j.rerank.model=/abs/path/to/reranker.gguf
```

In CI these reuse the project's existing shared GGUF cache (the chat, nomic-embedding and
jina-reranker models the core test jobs already download) — the
`test-java-llama-langchain4j-integration` job restores that cache and the
`Linux-x86_64` native library artifact, so no extra model is downloaded.

## Not mapped yet

- **Tool calling.** `ChatRequest.toolSpecifications()` are not forwarded, so the chat adapters return
  assistant *text*, not `AiMessage.toolExecutionRequests()`. (java-llama.cpp itself supports tool
  calling via `LlamaModel.chatWithTools` / typed `ToolDefinition`; bridging that to langchain4j
  `ToolSpecification` is the planned next step.)
- **Multimodal user input.** A multi-content `UserMessage` is flattened to its text parts; image/audio
  content is dropped.
- **Per-token tool-call / thinking stream events.** Streaming forwards plain text via
  `onPartialResponse`.
- **`response_format` (JSON mode).** `ChatRequest.responseFormat()` (json_object / json_schema) is not
  forwarded; `modelName()` is ignored since one model is bound per adapter.

Mapped request parameters: `temperature`, `topP`, `topK`, `maxOutputTokens`, `frequencyPenalty`,
`presencePenalty`, `stopSequences`. The non-streaming chat response carries the model's real finish
reason (`stop`/`length`/`tool_calls`) and token usage; the streaming completion carries assembled text
(no per-token usage).

Requires Java 17+ (langchain4j 1.x baseline). Targets `langchain4j-core` 1.17.1.
