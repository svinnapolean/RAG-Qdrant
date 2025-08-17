using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class EmbeddingService
{
    private readonly HttpClient _httpClient;

    public EmbeddingService()
    {
        _httpClient = new HttpClient { BaseAddress = new Uri("http://localhost:11434") }; // Ollama local API
    }

    public async Task<float[]> GenerateEmbeddingAsync(string text)
    {
        var payload = new { model = "phi4", input = text };
        var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("/embeddings", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<OllamaEmbeddingResponse>(json);

        return result?.Embedding ?? Array.Empty<float>();
    }

    //public Task<float[]> GeneratePhi4EmbeddingAsync(string text)
    //{
    //    // Assume you've preloaded the model and tokenizer
    //    var tokens = Tokenize(text); // Your tokenizer logic
    //    var inputTensor = CreateTensor(tokens); // Convert to tensor format

    //    using var session = new InferenceSession("phi4.onnx");
    //    var inputs = new List<NamedOnnxValue>
    //    {
    //        NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
    //    };

    //    using var results = session.Run(inputs);
    //    var embedding = results.First().AsEnumerable<float>().ToArray();

    //    return Task.FromResult(embedding);
    //}

    public async Task<float[]> GenerateHuggFaceEmbeddingAsync(string text)
    {
        var payload = new { inputs = text };
        var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

        _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "your_huggingface_token");

        var response = await _httpClient.PostAsync("https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        var embedding = JsonSerializer.Deserialize<float[]>(json); // Or wrap in a class if needed

        return embedding ?? Array.Empty<float>();
    }


}

public class EmbeddingGenerator
{
    private static InferenceSession _session;
    private readonly WordPieceTokenizer _tokenizer;

    public EmbeddingGenerator(string modelPath, string vocabPath)
    {
        _session = GetSession(modelPath);//new InferenceSession(modelPath);
        var vocab = new VocabLoader().Load(vocabPath);
        _tokenizer = new WordPieceTokenizer(vocab);
    }

    public static InferenceSession GetSession(string modelPath)
    {
        if (_session == null)
        {
            var options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            options.AppendExecutionProvider_CPU(useArena: 1);

            _session = new InferenceSession(modelPath, options);
        }
        return _session;
    }

    public float[] GenerateEmbedding(string text)
    {
        var (inputIds, attentionMask) = _tokenizer.Tokenize(text);

        var inputTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
        var attentionTensor = new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionTensor)
        };

        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        return output;
    }
}


public class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly string _unkToken = "[UNK]";
    private readonly string _clsToken = "[CLS]";
    private readonly string _sepToken = "[SEP]";

    public WordPieceTokenizer(Dictionary<string, int> vocab)
    {
        _vocab = vocab;
    }

    public (long[] inputIds, long[] attentionMask) Tokenize(string text)
    {
        var tokens = new List<int> { _vocab[_clsToken] };

        foreach (var word in text.Split(' '))
        {
            if (_vocab.ContainsKey(word))
                tokens.Add(_vocab[word]);
            else
                tokens.Add(_vocab[_unkToken]);
        }

        tokens.Add(_vocab[_sepToken]);

        var attention = Enumerable.Repeat(1L, tokens.Count).ToArray();
        return (tokens.Select(x => (long)x).ToArray(), attention);
    }
}



public class VocabLoader
{
    public Dictionary<string, int> Load(string path)
    {
        return File.ReadLines(path)
                   .Select((token, index) => new { token, index })
                   .ToDictionary(x => x.token, x => x.index);
    }
}


public class OllamaEmbeddingResponse
{
    public float[] Embedding { get; set; }
}

public class SearchResult
{
    public ulong Id { get; set; }
    public float Score { get; set; }
    public string Text { get; set; }
    public int RandNumber { get; set; }
}
