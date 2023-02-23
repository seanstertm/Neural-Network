package activation;

public class SiLU implements Activation {
    public double CalculateActivation(double[] inputs, int index) {
        return inputs[index] / (1 + Math.exp(-inputs[index]));
    }

    public double Derivative(double[] inputs, int index)
    {
        double sig = 1 / (1 + Math.exp(-inputs[index]));
        return inputs[index] * sig * (1 - sig) + sig;
    }

    public ActivationType GetActivationType()
    {
        return ActivationType.SiLU;
    }
}
