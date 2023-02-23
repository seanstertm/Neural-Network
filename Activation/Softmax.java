package activation;

public class Softmax implements Activation {
    public double CalculateActivation(double[] inputs, int index) {
        double expSum = 0;
        for (int i = 0; i < inputs.length; i++)
        {
            expSum += Math.exp(inputs[i]);
        }

        double res = Math.exp(inputs[index]) / expSum;

        return res;
    }

    public double Derivative(double[] inputs, int index) {
        double expSum = 0;
        for (int i = 0; i < inputs.length; i++)
        {
            expSum += Math.exp(inputs[i]);
        }

        double ex = Math.exp(inputs[index]);

        return (ex * expSum - ex * ex) / (expSum * expSum);
    }

    public ActivationType GetActivationType() {
        return ActivationType.Softmax;
    }
}
