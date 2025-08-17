using System;
using System.Collections;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.Memory;
using Qdrant.Client;
using Qdrant.Client.Grpc;



public class QdrantMemoryStore
{
    private readonly QdrantClient _client;
    private readonly string _collectionName;
    private readonly EmbeddingService _embeddingService = new EmbeddingService();
    public QdrantMemoryStore(string host, string collectionName)
    {
        _client = new QdrantClient(host);
        _collectionName = collectionName;
    }

    public async Task SaveInformationAsync(string collection, string key, string text, CancellationToken cancellationToken = default)
    {
        var embedding = await GenerateEmbedding(text); // Use Phi-4 / Ollama
                                                       // Create collection if it doesn't exist

        // Check if collection exists
        var existsResponse = await _client.CollectionExistsAsync(collection);

        if (existsResponse)
        {
            await _client.DeleteCollectionAsync(collection);
            await _client.CreateCollectionAsync(collection, new VectorParams { Size = 1536, Distance = Distance.Cosine });
        }
        /// Generate embeddings for each input text
        PointStruct pointStruct = new PointStruct() { Id = (ulong)GetTimestampId() , Vectors = ResizeVector(embedding,1536), Payload =  { ["text"] = text, ["length"] = text.Count(), ["category"] = "demo" } };


        var points = new List<PointStruct>();
        points.Add(pointStruct);
        await _client.UpsertAsync(_collectionName, points);
    }

    public float[] ResizeVector(float[] input, int targetSize = 1536)
    {
        if (input.Length == targetSize) return input;
        if (input.Length > targetSize) return input.Take(targetSize).ToArray();

        var padded = new float[targetSize];
        Array.Copy(input, padded, input.Length);
        return padded;
    }


    public static ulong GetTimestampId()
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        return (ulong)timestamp;
    }


    public async Task<List<SearchResult>> SearchAsync(string collection, string query, int limit = 5, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var queryEmbedding = await GenerateEmbedding(query);
        var results = await _client.SearchAsync(_collectionName, ResizeVector(queryEmbedding), limit:5);

        return results.Select(r => new SearchResult
        {
            Id = r.Id.Num,
            Score = r.Score,
            Text = r.Payload.TryGetValue("text", out Value? text) ? Convert.ToString(text.ToString()) : "",
            RandNumber = r.Payload.TryGetValue("rand_number", out var num) ? Convert.ToInt32(num) : -1
        }).ToList();

    }

    // You need a method to generate embeddings
    private Task<float[]> GenerateEmbedding(string text)
    {
        ///return _embeddingService.GenerateHuggFaceEmbeddingAsync(text);
        //string baseDir = AppDomain.CurrentDomain.BaseDirectory;

        //string model_path = Path.GetFullPath(Path.Combine(baseDir,"..", "..","onnx_model", "model.onnx"));
        //string vodab_path = Path.GetFullPath(Path.Combine(baseDir,"..", "..","onnx_model", "vocab.txt"));

        string model_folder = @"D:\\source\\onnx_model\\";
        var generator = new EmbeddingGenerator((model_folder + "model.onnx"), (model_folder + "vocab.txt"));
        return Task.Run(() => generator.GenerateEmbedding(text));

    }
}
