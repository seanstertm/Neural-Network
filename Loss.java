public interface Loss {
    double CalculateLoss(double[] outputs, double[] expectedOutputs);

    double Derivative(double outputs, double expectedOutputs);

    LossType GetLossType();
}
