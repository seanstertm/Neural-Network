package neuralnetwork;

import java.util.Random;

import activation.Activation;
import loss.Loss;
import run.CONFIG;

// The layer code contains the individual node wieghts and biases
// and functions to change and calculate those weights and biases
public class Layer {
    public int numNodesIn;
    public int numNodesOut;

    // Weights is practically a 2d array, but flattened to 1d
    // for performance and ease of iteration
    public double[] weights;
    public double[] biases;

    public double[] lossGradientWeights;
    public double[] lossGradientBiases;

    public double[] weightVelocities;
    public double[] biasVelocities;

    // Allows each layer to have an individual activation
    // mainly so the last layer can have softmax
    public Activation activation;

    public Layer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        
        activation = CONFIG.activationType;

        weights = new double[numNodesIn * numNodesOut];
        biases = new double[numNodesOut];
        lossGradientWeights = new double[numNodesIn * numNodesOut];
        lossGradientBiases = new double[numNodesOut];
        weightVelocities = new double[numNodesIn * numNodesOut];
        biasVelocities = new double[numNodesOut];

        InitializeRandomWeights();
    }

    // Calculates the layer outputs
    public double[] CalculateOutputs(double[] inputs) {

        // Runs the inputs through the network, getting each output
        double[] weightedInputs = new double[numNodesOut];
        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];
            for(int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput;
        }

        // Apply the activation function to each input
        double[] activations = new double[numNodesOut];
        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            activations[nodeOut] = activation.CalculateActivation(weightedInputs, nodeOut);
        }
        return activations;
    }

    // Overload with the learn data
    // Data gets stored in the learn data object
    public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData) {
        learnData.inputs = inputs;
        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];
            for(int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            learnData.weightedInputs[nodeOut] = weightedInput;
        }

        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            learnData.activations[nodeOut] = activation.CalculateActivation(learnData.weightedInputs, nodeOut);
        }
        return learnData.activations;
    }

    // This calculates the partial derivatives and updates the gradients
    // This technique is called gradient descent and uses
    // derivatives to minimize the loss function
    public void UpdateGradients(LayerLearnData layerLearnData)
	{
        // Update gradients by using node values
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double nodeValue = layerLearnData.nodeValues[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
                lossGradientWeights[GetWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
            }
        }

        // Update gradients for biases
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double derivativeCostBias = layerLearnData.nodeValues[nodeOut];
            lossGradientBiases[nodeOut] += derivativeCostBias;
        }
	}

    // Set the node values of the layer data object for the last layer
    public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, Loss loss)
	{
		for (int i = 0; i < layerLearnData.nodeValues.length; i++)
		{
			double costDerivative = loss.Derivative(layerLearnData.activations[i], expectedOutputs[i]);
			double activationDerivative = activation.Derivative(layerLearnData.weightedInputs, i);
			layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
		}
	}

    // Sets the node values of the hidden layer using the last layer's derivative
    public void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer oldLayer, double[] oldNodeValues)
	{
		for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++)
		{
			double newNodeValue = 0;
			for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++)
			{
				double weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
				newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
			}
			newNodeValue *= activation.Derivative(layerLearnData.weightedInputs, newNodeIndex);
			layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
		}

	}

    // Applies the gradient arrays to the weights and biases with other parameters
    // Also resets the gradient to 0
    public void ApplyGradients(double learnRate, double regularization, double momentum)
	{
		double weightDecay = (1 - regularization * learnRate);

		for (int i = 0; i < weights.length; i++)
		{
			double weight = weights[i];
			double velocity = weightVelocities[i] * momentum - lossGradientWeights[i] * learnRate;
			weightVelocities[i] = velocity;
			weights[i] = weight * weightDecay + velocity;
			lossGradientWeights[i] = 0;
		}

		for (int i = 0; i < biases.length; i++)
		{
			double velocity = biasVelocities[i] * momentum - lossGradientBiases[i] * learnRate;
			biasVelocities[i] = velocity;
			biases[i] += velocity;
			lossGradientBiases[i] = 0;
		}
	}

    // Returns the weight given [][]
    public double GetWeight(int nodeIn, int nodeOut) {
        return weights[GetWeightIndex(nodeIn, nodeOut)];
    }

    // Since weights is a 1d array, translates [][] to []
    public int GetWeightIndex(int nodeIn, int nodeOut) {
        return nodeOut * numNodesIn + nodeIn;
    }

    // Creates random weights based on the normal curve
    private void InitializeRandomWeights() {
        Random rng = new Random();

        for(int i = 0; i < weights.length; i++) {
            // formula for normal curve distribution
            weights[i] = (Math.sqrt(-2.0 * Math.log(1 - rng.nextDouble())) * Math.cos(2.0 * Math.PI * (1 - rng.nextDouble()))) / Math.sqrt(numNodesIn);
        }
    }
}
