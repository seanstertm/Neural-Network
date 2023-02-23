package activation;

// Interface to handle all activation functions
public interface Activation {
    double CalculateActivation(double[] inputs, int index);
    
    double Derivative(double[] inputs, int index);

    ActivationType GetActivationType();
}
