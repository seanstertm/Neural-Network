package loss;

// Interface to handle all loss types
public interface Loss {
    // This function is not used, but is kept for debugging sake
    double CalculateLoss(double[] outputs, double[] expectedOutputs);

    double Derivative(double outputs, double expectedOutputs);

    LossType GetLossType();
}
