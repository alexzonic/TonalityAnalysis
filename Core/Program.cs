using System;
using System.IO;
using Analyzer;

namespace Core
{
    public sealed class Program
    {
        public static void Main(string[] args)
        {
            IDataAnalyzer dataAnalyzer =
                new DataAnalyzer($"{Path.GetTempPath()}training.1600000.processed.noemoticon2.csv", AnalyzeType.Sentiment);

            var exampleMessage1 = "That's a great idea. It should work.";
            var exampleMessage2 = "free medicine winner! congratulations";
            var exampleMessage3 = "Yes we should meet over the weekend!";
            var exampleMessage4 = "you win pills and free entry vouchers";

            dataAnalyzer.AnalyzeEmotion(exampleMessage1);
            dataAnalyzer.AnalyzeEmotion(exampleMessage2);
            dataAnalyzer.AnalyzeEmotion(exampleMessage3);
            dataAnalyzer.AnalyzeEmotion(exampleMessage4);

            while (Continue())
            {
                dataAnalyzer.AnalyzeEmotion(Console.ReadLine());
            }
        }

        private static bool Continue()
        {
            Console.WriteLine("Continue tests? (y/n)");
            return Console.ReadLine() != "n";
        }
    }
}