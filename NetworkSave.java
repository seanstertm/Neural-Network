// These are not real JSONs, only my files can interpret these numbers
// This is done because codehs does not support external libraries

import java.util.Arrays;
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

    public static void SaveNetwork(NeuralNetwork network) {
        try{
            FileWriter writer = new FileWriter("network.txt");
            writer.write(StringifyNetwork(network));
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static NeuralNetwork LoadNetwork() {
        try {
            byte[] encoded = Files.readAllBytes(Paths.get("network.txt"));
            String stringifiedNetwork = new String(encoded);
            System.out.println(stringifiedNetwork);
        } catch (Exception e) {
            e.printStackTrace();
        }

        NeuralNetwork network = new NeuralNetwork();

        return network;
    }

    // Double.parseDouble()
}
