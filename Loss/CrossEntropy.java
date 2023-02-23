package loss;

// https://en.wikipedia.org/wiki/Cross_entropy
public class CrossEntropy implements Loss {
    public double CalculateLoss(double[] outputs, double[] expectedOutputs) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++)
        {
            double x = outputs[i];
            double y = expectedOutputs[i];
            double v = (y == 1) ? -Math.log(x) : -Math.log(1 - x);
            loss += Double.isNaN(v) ? 0 : v;
        }
        return loss;
    }

    public double Derivative(double outputs, double expectedOutputs) {
        if (outputs == 0 || outputs == 1)
        {
            return 0;
        }
        return (-outputs + expectedOutputs) / (outputs * (outputs - 1));
    }

    public LossType GetLossType() {
        return LossType.CrossEntropy;
    }
}
