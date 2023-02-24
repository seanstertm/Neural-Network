package neuralnetwork;

import activation.Softmax;
import datacontrol.DataPoint;
import datacontrol.NetworkOutput;
import loss.Loss;
import run.CONFIG;

// This is the only object that needs to be manually created
// This will set up a neural network and contains all user methods
public class NeuralNetwork {
    public Layer[] layers;
    public int[] layerSizes;

    public Loss loss;
    public NetworkLearnData[] batchLearnData;

    // The ... allows all parameters to be condensed into an int[]
    // This constructor creates one less layer than listed
    // Because the last int is the number of outputs
    public NeuralNetwork(int... layerSizes) {
        this.layerSizes = layerSizes;

        layers = new Layer[layerSizes.length - 1];
        for(int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }

        // Sets the last layer's activation to softmax for probability
        layers[layers.length - 1].activation = new Softmax();

        loss = CONFIG.lossType;
    }

    // This function combines multiple functions for the entire learning process of a batch
    public void Learn(DataPoint[] trainingData) {

        // This will create a new data object if the size of the previous object does not match
        if(batchLearnData == null || batchLearnData.length != trainingData.length) {
            batchLearnData = new NetworkLearnData[trainingData.length];
            for(int i = 0; i < batchLearnData.length; i++)
			{
				batchLearnData[i] = new NetworkLearnData(layers);
			}
        }

        // Updates gradients for each DataPoint
        for(int i = 0; i < trainingData.length; i++) {
            UpdateGradients(trainingData[i], batchLearnData[i]);
        }

        // Applies each layer's gradient
        for(int i = 0; i < layers.length; i++) {
            layers[i].ApplyGradients(CONFIG.learnRate/trainingData.length, CONFIG.regularization, CONFIG.momentum);
        }
    }

    // Calculation for gradient descent
    public void UpdateGradients(DataPoint data, NetworkLearnData learnData) {
        double[] nextLayerInputs = data.inputs;

        // Runs the data's inputs through the network
        for(int i = 0; i < layers.length; i++) {
            // This override of the calculate outputs function will save the data in the learnData objects
            nextLayerInputs = layers[i].CalculateOutputs(nextLayerInputs, learnData.layerData[i]);
        }

        // Updates gradients for all layers
        // Hidden Layers are done separately since the calculation is different
        int outputLayerIndex = layers.length - 1;
        Layer outputLayer = layers[outputLayerIndex];
        LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];

        outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs, loss);
        outputLayer.UpdateGradients(outputLearnData);

        // This is called backpropogation
        // Since each layer relies on the previous layer's partial derivatives,
        // each calculation only needs to be done once
        for(int i = outputLayerIndex - 1; i >= 0; i--) {
            LayerLearnData layerLearnData = learnData.layerData[i];
            Layer hiddenLayer = layers[i];

            hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, layers[i + 1], learnData.layerData[i + 1].nodeValues);
            hiddenLayer.UpdateGradients(layerLearnData);
        }
    }

    // Combines the following two functions into one output object
    public NetworkOutput Classify(double[] inputs) {
        double[] outputs = CalculateOutputs(inputs);
        int predictedOutput = MaxValueIndex(outputs);
        return new NetworkOutput(outputs, predictedOutput);
    }

    // Takes inputs and runs it through the network
    // Feeds the inputs of one layer into the next
    public double[] CalculateOutputs(double[] inputs) {
        for(Layer layer : layers) {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    // Simple function to return the index of the highest output
    public int MaxValueIndex(double[] outputs) {
        double maxValue = Double.MIN_VALUE;
		int index = 0;
		for (int i = 0; i < outputs.length; i++)
		{
			if (outputs[i] > maxValue)
			{
				maxValue = outputs[i];
				index = i;
			}
		}
		return index;
    }
}
