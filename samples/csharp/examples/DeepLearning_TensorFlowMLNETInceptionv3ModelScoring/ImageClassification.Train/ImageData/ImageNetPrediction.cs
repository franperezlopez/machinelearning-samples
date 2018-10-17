﻿using Microsoft.ML.Runtime.Api;
using System;

namespace ImageClassification.ImageData
{
    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }

    public class ImageNetStaticPrediction
    {
        public float[] Score;

        public string PredictedLabel;
    }

    public class ImageNetWithLabelStaticPrediction : ImageNetStaticPrediction
    {
        public ImageNetWithLabelStaticPrediction(ImageNetStaticPrediction pred, string label)
        {
            Label = label;
            Score = pred.Score;
            PredictedLabel = pred.PredictedLabel;
        }

        public string Label;
    }

}
