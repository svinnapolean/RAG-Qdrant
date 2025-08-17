
# 🧠 Agentic AI RAG-Qdrant: Semantic Kernel + Phi-4 + Qdrant

This program demonstrates a local Retrieval-Augmented Generation (RAG) pipeline using:

- **Semantic Kernel** for orchestration
- **Phi-4** via Ollama for chat completion
- **Qdrant** for vector search
- **Ollama embeddings** for document indexing

---

## ⚙️ Configuration

```csharp
// Ollama endpoint
private static readonly Uri OllamaEndpoint = new("http://localhost:11434/v1");
private const string OllamaApiKey = "ollama"; // dummy key

// Models
private const string ChatModel = "phi4";
private const string EmbeddingModel = "nomic-embed-text";

// Qdrant setup
private const string QdrantEndpoint = "localhost";
private const string Collection = "rag_dotnet_docs";
private const int EmbeddingDimension = 768;
```

---

## 🧱 Pipeline Steps

### 1. **Build Kernel**
```csharp
builder.AddOpenAIChatCompletion(...);
builder.Services.AddSingleton(new OllamaEmbeddingGenerator("nomic-embed-text"));
var qdrantMemory = new QdrantMemoryStore(QdrantEndpoint, Collection);
```

- Connects to local Phi-4 via OpenAI-compatible API
- Uses Ollama for embeddings
- Initializes Qdrant vector store

---

### 2. **Ingest Documents**
```csharp
var seedDocs = new[] {
  ("doc1", "Semantic Kernel is a Microsoft framework..."),
  ("doc2", "Phi-4 is a local-friendly open model..."),
  ("doc3", "Qdrant is a vector database...")
};
foreach (var (id, text) in seedDocs)
    await qdrantMemory.SaveInformationAsync(Collection, key: id, text: text);
```

- Indexes demo documents into Qdrant
- Replace with file ingestion for production

---

### 3. **Search for Relevant Context**
```csharp
var results = qdrantMemory.SearchAsync(Collection, question, limit: 5);
```

- Retrieves top-k relevant chunks
- Filters by similarity score

---

### 4. **Prompt Phi-4 with Context**
```csharp
var system = """You are a precise, helpful assistant...""";
var user = $"""Question: {question} Context snippets: {contextBlock}...""";
var response = await chat.GetChatMessageContentAsync(history);
```

- Constructs a grounded prompt using retrieved context
- Enforces citation and factual accuracy

---

## 📤 Output Format

```plaintext
=== RAG Answer ===
Answer:
• ...

Citations: [#1], [#2], ...

--- Retrieved ---
[#1] id: doc1 score: 0.923
[#2] id: doc2 score: 0.887
```

---

## 🧪 Notes

- You can replace `seedDocs` with file ingestion or web scraping.
- Adjust `minScore` and `topK` to tune retrieval sensitivity.
- Extend with Semantic Kernel planner or memory modules for agentic workflows.

---
