package NeuralNetwork;

import java.util.Random;

import Activation.Activation;
import Loss.Loss;
import Run.CONFIG;

public class Layer {
    public int numNodesIn;
    public int numNodesOut;

    public double[] weights;
    public double[] biases;

    public double[] lossGradientWeights;
    public double[] lossGradientBiases;

    public double[] weightVelocities;
    public double[] biasVelocities;

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

    public double[] CalculateOutputs(double[] inputs) {
        double[] weightedInputs = new double[numNodesOut];
        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];
            for(int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput;
        }

        double[] activations = new double[numNodesOut];
        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            activations[nodeOut] = activation.CalculateActivation(weightedInputs, nodeOut);
        }
        return activations;
    }

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

    public void UpdateGradients(LayerLearnData layerLearnData)
	{
			for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
			{
				double nodeValue = layerLearnData.nodeValues[nodeOut];
				for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
				{
					double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
					lossGradientWeights[GetWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
				}
			}

			for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
			{
				double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
				lossGradientBiases[nodeOut] += derivativeCostWrtBias;
			}
	}

    public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, Loss loss)
	{
		for (int i = 0; i < layerLearnData.nodeValues.length; i++)
		{
			double costDerivative = loss.Derivative(layerLearnData.activations[i], expectedOutputs[i]);
			double activationDerivative = activation.Derivative(layerLearnData.weightedInputs, i);
			layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
		}
	}

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

    public double GetWeight(int nodeIn, int nodeOut) {
        int flattenedIndex = nodeOut * numNodesIn + nodeIn;
        return weights[flattenedIndex];
    }

    public int GetWeightIndex(int nodeIn, int nodeOut) {
        return nodeOut * numNodesIn + nodeIn;
    }

    private void InitializeRandomWeights() {
        Random rng = new Random();

        for(int i = 0; i < weights.length; i++) {
            // formula for normal curve distribution
            weights[i] = (Math.sqrt(-2.0 * Math.log(1 - rng.nextDouble())) * Math.cos(2.0 * Math.PI * (1 - rng.nextDouble()))) / Math.sqrt(numNodesIn);
        }
    }
}
