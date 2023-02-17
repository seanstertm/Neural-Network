package Activation;

public class Sigmoid implements Activation {
    public double CalculateActivation(double[] inputs, int index)
    {
        return 1.0 / (1 + Math.exp(-inputs[index]));
    }

    public double Derivative(double[] inputs, int index)
    {
        double a = CalculateActivation(inputs, index);
        return a * (1 - a);
    }

    public ActivationType GetActivationType()
    {
        return ActivationType.Sigmoid;
    }
}
