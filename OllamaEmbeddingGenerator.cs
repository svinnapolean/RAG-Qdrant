using System;
using System.Collections.Generic;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

public class OllamaEmbeddingGenerator 
{
    private readonly HttpClient _http;
    private readonly string _model;

    public OllamaEmbeddingGenerator(string model = "phi4:latest")
    {
        _http = new HttpClient { BaseAddress = new Uri("http://localhost:11434") };
        _model = model;
    }

    public async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<ReadOnlyMemory<float>>();

        foreach (var text in data)
        {
            var request = new
            {
                model = _model,
                prompt = text
            };

            using var response = await _http.PostAsJsonAsync("/api/embeddings", request, cancellationToken);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync(cancellationToken);

            using var doc = JsonDocument.Parse(json);
            var embedding = doc.RootElement
                .GetProperty("embedding")
                .EnumerateArray()
                .Select(x => x.GetSingle())
                .ToArray();

            results.Add(new ReadOnlyMemory<float>(embedding));
        }

        return results;
    }
}
