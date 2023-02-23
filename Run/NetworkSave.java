package run;

import java.util.Arrays;

import neuralnetwork.NeuralNetwork;

import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

public class NetworkSave {
    public static String StringifyNetwork(NeuralNetwork network) {
        String string = Arrays.toString(network.layerSizes);
        for(int i = 0; i < network.layers.length; i++) {
            string += Arrays.toString(network.layers[i].weights) + Arrays.toString(network.layers[i].biases);
        }
        return string;
    }

    public static NeuralNetwork NetworkFromString(String string) {
        String[] arrays = string.split("\\]\\[");

        // Strip leading and trailing brackets
        arrays[0] = arrays[0].substring(1);
        arrays[arrays.length - 1] = arrays[arrays.length - 1].substring(0, arrays[arrays.length - 1].length() - 1);

        String[] networkSizes = arrays[0].split(", ");
        int[] layerSizes = new int[networkSizes.length];
        for(int i = 0; i < networkSizes.length; i++) {
            layerSizes[i] = Integer.parseInt(networkSizes[i]);
        }

        NeuralNetwork network = new NeuralNetwork(layerSizes);

        for(int i = 0; i < layerSizes.length - 1; i++) {
            String[] stringWeights = arrays[i * 2 + 1].split(", ");
            double[] weights = new double[stringWeights.length];
            for(int j = 0; j < stringWeights.length; j++) {
                weights[j] = Double.parseDouble(stringWeights[j]);
            }
            network.layers[i].weights = weights;

            String[] stringBiases = arrays[i * 2 + 2].split(", ");
            double[] biases = new double[stringBiases.length];
            for(int j = 0; j < stringBiases.length; j++) {
                biases[j] = Double.parseDouble(stringBiases[j]);
            }
            network.layers[i].biases = biases;
        }

        return network;
    }

    public static void SaveNetwork(NeuralNetwork network) {
        try{
            FileWriter writer = new FileWriter("Run/network.txt");
            writer.write(StringifyNetwork(network));
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static NeuralNetwork LoadNetwork() {
        try {
            byte[] encoded = Files.readAllBytes(Paths.get("Run/network.txt"));
            return NetworkFromString(new String(encoded));
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // Double.parseDouble()
}
