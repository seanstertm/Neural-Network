package activation;

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
public class ReLU implements Activation {
    public double CalculateActivation(double[] inputs, int index)
    {
        return Math.max(0, inputs[index]);
    }

    public double Derivative(double[] inputs, int index)
    {
        return inputs[index] > 0 ? 1 : 0;
    }

    public ActivationType GetActivationType()
    {
        return ActivationType.ReLU;
    }
}
