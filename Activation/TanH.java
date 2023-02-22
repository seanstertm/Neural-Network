package activation;

public class TanH implements Activation {
    public double CalculateActivation(double[] inputs, int index)
    {
        double e = Math.exp(2 * inputs[index]);
        return (e - 1) / (e + 1);
    }

    public double Derivative(double[] inputs, int index)
    {
        double a = CalculateActivation(inputs, index);
        return 1 - a * a;
    }

    public ActivationType GetActivationType()
    {
        return ActivationType.TanH;
    }
}
