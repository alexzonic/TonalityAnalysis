using System;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace Analyzer
{
    public sealed class DataAnalyzer : IDataAnalyzer
    {
        private readonly MLContext _mlContext;
        private readonly PredictionEngine<SentimentIssue, SentimentPrediction> _emotionsPredictor;

        public DataAnalyzer(string pathToData, AnalyzeType analyzeType)
        {
            _mlContext = new MLContext();
            if (analyzeType == AnalyzeType.Sentiment)
                _emotionsPredictor = TrainAnalyzeEmotions(pathToData);
        }

        public void AnalyzeEmotion(string message)
        {
            if (_emotionsPredictor == null)
            {
                Console.WriteLine("System not trained analyze emotions yet");
                return;
            }

            var input = new SentimentIssue {Message = message};
            var prediction = _emotionsPredictor.Predict(input);

            prediction.Prediction = prediction.Score.GetItemOrDefault(1) * 4f;
            Console.WriteLine($"The tonality for message '{input.Message}' is {prediction.Prediction} [0, 4]");
        }

        private PredictionEngine<SentimentIssue, SentimentPrediction> TrainAnalyzeEmotions(string pathToData)
        {
            var dataView = _mlContext.Data.LoadFromTextFile<SentimentIssue>(pathToData, hasHeader: true, separatorChar: ',');

            var trainTestSplit = _mlContext.Data.TrainTestSplit(dataView, 0.2);
            var trainingData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            var dataProcessPipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText("FeaturesText", new TextFeaturizingEstimator.Options
                {
                    WordFeatureExtractor = new WordBagEstimator.Options
                    {
                        NgramLength = 2,
                        UseAllLengths = true
                    },
                    CharFeatureExtractor = new WordBagEstimator.Options
                    {
                        NgramLength = 2,
                        UseAllLengths = false,
                    },
                    Norm = TextFeaturizingEstimator.NormFunction.L2,
                }, "Message"))
                .Append(_mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                .AppendCacheCheckpoint(_mlContext);

            var trainer = _mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    _mlContext.BinaryClassification.Trainers.AveragedPerceptron())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeLine = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeLine.Fit(trainingData);
            var predictions = trainedModel.Transform(testData);

            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults =
                _mlContext.MulticlassClassification.CrossValidate(predictions, trainingPipeLine);


            ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            return _mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);
        }
    }
}