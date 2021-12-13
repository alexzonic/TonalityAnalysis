using Microsoft.ML.Data;

namespace Analyzer
{
    internal sealed class SentimentIssue
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(5)]
        public string Message { get; set; }
    }
}