using System;
using System.Linq;
using ImageClassification.ImageData;
using System.IO;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using static ImageClassification.Model.ConsoleHelpers;

namespace ImageClassification.Model
{
    public class ModelEvaluator
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly IHostEnvironment env;

        public ModelEvaluator(string dataLocation, string imagesFolder, string modelLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            env = new LocalEnvironment();
        }

        public void EvaluateStaticApi()
        {
            ConsoleWriteHeader("Loading model");
            Console.WriteLine($"Model loaded: {modelLocation}");

            ITransformer loadedModel;
            using (var f = new FileStream(modelLocation, FileMode.Open))
                loadedModel = TransformerChain.LoadFrom(env, f);

            var predictor = loadedModel.MakePredictionFunction<ImageNetData, ImageNetStaticPrediction>(env);
            var testData = ImageNetData.ReadFromCsv(dataLocation, imagesFolder).ToList();

            ConsoleWriteHeader("Making classifications");
            // There is a bug (), that always buffers the response from the predictor
            // so we have to make a copy-by-value op everytime we get a response
            // from the predictor
            testData
                .Select(td => new { td, pred = predictor.Predict(td) })
                .Select(pr => (pr.td.ImagePath, pr.pred.PredictedLabel, pr.pred.Score))
                .ToList()
                .ForEach(pr => ConsoleWriteImagePrediction(pr.ImagePath, pr.PredictedLabel, pr.Score.Max()));
        }
    }
}
