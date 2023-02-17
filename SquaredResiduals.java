public class SquaredResiduals implements Loss {
    public double CalculateLoss(double[] predictedOutputs, double[] expectedOutputs)
    {
        double loss = 0;
        for (int i = 0; i < predictedOutputs.length; i++)
        {
            double error = predictedOutputs[i] - expectedOutputs[i];
            loss += error * error;
        }
        return 0.5 * loss;
    }

    public double Derivative(double predictedOutput, double expectedOutput)
    {
        return predictedOutput - expectedOutput;
    }

    public LossType GetLossType()
    {
        return LossType.MeanSquareError;
    }
}
