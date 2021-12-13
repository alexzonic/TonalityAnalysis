using System;
using Microsoft.ML.Data;

namespace Analyzer
{
    internal sealed class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public float Prediction { get; set; }

        public float Probability { get; set; }

        public VBuffer<float> Score { get; set; }
    }
}