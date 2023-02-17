package activation;

public interface Activation {
    double CalculateActivation(double[] inputs, int index);
    
    double Derivative(double[] inputs, int index);

    ActivationType GetActivationType();
}
