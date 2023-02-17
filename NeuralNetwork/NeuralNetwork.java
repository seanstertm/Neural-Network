package NeuralNetwork;

import DataControl.DataPoint;
import DataControl.NetworkOutput;
import Loss.Loss;
import Run.CONFIG;

public class NeuralNetwork {
    public Layer[] layers;
    public int[] layerSizes;

    public Loss loss;
    public NetworkLearnData[] batchLearnData;

    public NeuralNetwork(int... layerSizes) {
        this.layerSizes = layerSizes;

        layers = new Layer[layerSizes.length - 1];
        for(int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }

        loss = CONFIG.lossType;
    }

    public void Learn(DataPoint[] trainingData) {
        if(batchLearnData == null || batchLearnData.length != trainingData.length) {
            batchLearnData = new NetworkLearnData[trainingData.length];
            for(int i = 0; i < batchLearnData.length; i++)
			{
				batchLearnData[i] = new NetworkLearnData(layers);
			}
        }

        for(int i = 0; i < trainingData.length; i++) {
            UpdateGradients(trainingData[i], batchLearnData[i]);
        }

        for(int i = 0; i < layers.length; i++) {
            layers[i].ApplyGradients(CONFIG.learnRate/trainingData.length, CONFIG.regularization, CONFIG.momentum);
        }
    }

    public void UpdateGradients(DataPoint data, NetworkLearnData learnData) {
        double[] nextLayerInputs = data.inputs;

        for(int i = 0; i < layers.length; i++) {
            nextLayerInputs = layers[i].CalculateOutputs(nextLayerInputs, learnData.layerData[i]);
        }

        int outputLayerIndex = layers.length - 1;
        Layer outputLayer = layers[outputLayerIndex];
        LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];

        outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs, loss);
        outputLayer.UpdateGradients(outputLearnData);

        for(int i = outputLayerIndex - 1; i >= 0; i--) {
            LayerLearnData layerLearnData = learnData.layerData[i];
            Layer hiddenLayer = layers[i];

            hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, layers[i + 1], learnData.layerData[i + 1].nodeValues);
            hiddenLayer.UpdateGradients(layerLearnData);
        }
    }

    public NetworkOutput Classify(double[] inputs) {
        double[] outputs = CalculateOutputs(inputs);
        int predictedOutput = MaxValueIndex(outputs);
        return new NetworkOutput(outputs, predictedOutput);
    }

    public double[] CalculateOutputs(double[] inputs) {
        for(Layer layer : layers) {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

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
