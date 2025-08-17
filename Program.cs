using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Connectors.Qdrant;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;

internal class Program
{
    // ---------- CONFIG ----------
    // Ollama OpenAI-compatible endpoint
    private static readonly Uri OllamaEndpoint = new("http://localhost:11434/v1");
    private const string OllamaApiKey = "ollama"; // dummy

    // Models
    private const string ChatModel = "phi4";                 // generation
    private const string EmbeddingModel = "nomic-embed-text"; // 768-d embeddings

    // Qdrant
    private const string QdrantEndpoint = "localhost";
    private const int EmbeddingDimension = 768; // must match 'nomic-embed-text'

    // Collections
    private const string Collection = "rag_dotnet_docs";

    private static async Task Main(string[] args)
    {
        // =============== 1) Build Kernel ===============
        var builder = Kernel.CreateBuilder();

        // Local Phi-4 via OpenAI-compatible connector
        builder.AddOpenAIChatCompletion(
            modelId: ChatModel,
            endpoint: OllamaEndpoint,
            apiKey: OllamaApiKey);

        // Local embeddings via Ollama (OpenAI-compatible /v1/embeddings)
        // Use Ollama for embeddings
        builder.Services.AddSingleton(
            new OllamaEmbeddingGenerator("nomic-embed-text") // Or phi4 if it supports embeddings
        );

        // Qdrant vector store
        var qdrantMemory = new QdrantMemoryStore(QdrantEndpoint, Collection);
   
        var kernel = builder.Build();
        var chat = kernel.GetRequiredService<IChatCompletionService>();

        // =============== 2) Ingest Documents ===============
        // Option A: quick demo strings (remove if you ingest files)
        var seedDocs = new[]
        {
            ("doc1", "Semantic Kernel is a Microsoft framework to orchestrate LLMs, tools, and memories."),
            ("doc2", "Phi-4 is a local-friendly open model; combine it with RAG for grounded answers."),
            ("doc3", "Qdrant is a vector database used to store and search embeddings efficiently.")
        };

        Console.WriteLine("Indexing demo knowledge into Qdrant...");
        foreach (var (id, text) in seedDocs)
            await qdrantMemory.SaveInformationAsync(Collection, key: id, text: text);

        Console.WriteLine("Ingestion complete.\n");

        // =============== 3) Ask a question ===============
        var question = args.Length > 0
            ? string.Join(' ', args)
            : "How do I use Semantic Kernel with local Phi-4 and Qdrant to build RAG?";

        // Retrieve top-k relevant chunks
        var topK = 5;
        var minScore = 0.5; // 0..1 (higher is more similar)
        var results = qdrantMemory.SearchAsync(Collection, question, limit: topK);

        var sbContext = new StringBuilder();
        var sbCitations = new StringBuilder();
        int rank = 1;

        foreach (var r in results.Result)
        {
            // r.Metadata.Id (if available) is the original id you saved
            var id = r.Id.ToString() ?? $"doc_{rank}";
            var text = r.Text ?? "";
            sbContext.AppendLine($"[#{rank}] {text}");
            sbCitations.AppendLine($"[#{rank}] id: {id}  score: {r.Score:0.000}");
            rank++;
        }

        var contextBlock = sbContext.ToString().Trim();
        if (string.IsNullOrWhiteSpace(contextBlock))
        {
            Console.WriteLine("No relevant context found. Try lowering minScore or indexing more docs.");
            return;
        }

        // =============== 4) Prompt Phi-4 with retrieved context ===============
        var system = """
        You are a precise, helpful assistant.
        Answer ONLY using the provided context. If missing info, say you don't know.
        Always include a "Citations" section listing the snippet numbers you used.
        """;

        var user = $"""
        Question:
        {question}

        Context snippets:
        {contextBlock}

        Instructions:
        - Use the snippets to compose a concise, accurate answer.
        - Do not invent facts.
        - Provide bullet-point steps when helpful.

        Reply format:
        Answer:
        • ...

        Citations:
        [#1], [#2], ...
        """;

        var history = new ChatHistory();
        history.AddSystemMessage(system);
        history.AddUserMessage(user);

        var response = await chat.GetChatMessageContentAsync(history);

        Console.WriteLine("\n=== RAG Answer ===\n");
        Console.WriteLine(response.Content?.Trim());
        Console.WriteLine("\n--- Retrieved ---");
        Console.WriteLine(sbCitations.ToString());
    }

    private static string NormalizeWhitespace(string s)
        => Regex.Replace(s, @"\s+", " ").Trim();
}