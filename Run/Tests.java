package run;

import java.io.File;
import java.io.FileInputStream;
import java.util.*;

import datacontrol.Batch;
import datacontrol.DataHandler;
import datacontrol.DataPoint;
import datacontrol.NetworkOutput;
import neuralnetwork.NeuralNetwork;

public class Tests {
    public static void main(String[] args) {
        Stringify();
    }

    public static void Mnist() {
        DataPoint[] data = ReadMnist(true);

        NeuralNetwork network = new NeuralNetwork(784, 300, 10);

        for(int epoch = 0; epoch < 2000; epoch++) {
            DataHandler.splitData(data, 100, 0.8);

            for(Batch batch : DataHandler.trainingBatches) {
                network.Learn(batch.data);
            }

            int correct = 0;
            for(DataPoint dataPoint : DataHandler.testData) {
                NetworkOutput output = network.Classify(dataPoint.inputs);
                if(dataPoint.expectedOutputs[output.predictedClass] == 1) {
                    correct++;
                }
            }

            System.out.println("Epoch: " + epoch + " Test accuracy: " + 100.0 * correct / DataHandler.testData.length + "%");
        }

        System.out.println("\n\n\n\n\n----------------\n\n\n\n\n");

        for(DataPoint dataPoint : DataHandler.testData) {
            NetworkOutput output = network.Classify(dataPoint.inputs);
            System.out.println("Image: " + dataPoint.label + "  Guess: " + output.predictedClass + "  Confidence: " + output.outputs[output.predictedClass]);
        }
    }

    public static DataPoint[] ReadMnist(boolean test) {
        try {
            int size = test ? 1000 : 6000;

            File file = new File(test ? "Data/mnistTestLabels.idx1-ubyte" : "Data/mnistTrainLabels.idx1-ubyte");
            byte[] contents = new byte[(int)file.length()];
            FileInputStream input = new FileInputStream(file);
            input.read(contents);

            int[] labels = new int[(int)file.length() - 16];

            for(int i = 16; i < file.length(); i++) {
                labels[i - 16] = contents[i];
            }

            input.close();

            file = new File(test ? "Data/mnistTest.idx3-ubyte" : "Data/mnistTrain.idx3-ubyte");
            contents = new byte[(int)file.length()];
            input = new FileInputStream(file);
            input.read(contents);
            
            DataPoint[] data = new DataPoint[size];
            int i = 16;
            for(int image = 0; image < size; image++) {
                double[] inputs = new double[28*28];
                for(int pixelCount = 0; pixelCount < 28 * 28; pixelCount++) {
                    int pixel = contents[i];
                    if(pixel < 0) { pixel += 256; }
                    inputs[pixelCount] = pixel / 256.0;
                    i++;
                }
                data[image] = new DataPoint(inputs, labels[image], 10);
            }
            input.close();
            return data;
            
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void Stringify() {
        NeuralNetwork network = NetworkSave.LoadNetwork();

        System.out.println(network);
    }

    public static void SmallerOfTwoNumbersTest() {
        Random rng = new Random();

        Scanner input = new Scanner(System.in);

        NeuralNetwork network = new NeuralNetwork(2, 3, 2);

        for(int epoch = 0; epoch < 50; epoch++) {
            DataPoint[] randomData = new DataPoint[50000];
            for(int i = 0; i < randomData.length; i++) {
                double x = rng.nextDouble();
                double y = rng.nextDouble();
                randomData[i] = new DataPoint(new double[]{x, y}, x > y ? 1 : 0, 2);
            }
            DataHandler.splitData(randomData, 50, 0.8);

            for(int i = 0; i < DataHandler.trainingBatches.length; i++) {
                network.Learn(DataHandler.trainingBatches[i].data);
            }

            int correct = 0;

            for(int i = 0; i < DataHandler.testData.length; i++) {
                NetworkOutput output = network.Classify(DataHandler.testData[i].inputs);
                if(DataHandler.testData[i].expectedOutputs[output.predictedClass] ==  1) {
                    correct++;
                }
            }

            System.out.println(100.0 * correct / DataHandler.testData.length + "% accuracy for testing data");
        }

        double[] inputs = new double[2];
        System.out.print("Finished Training\n\nEnter a double: ");
        inputs[0] = input.nextDouble();
        System.out.print("Enter another double: ");
        inputs[1] = input.nextDouble();

        input.close();

        NetworkOutput output = network.Classify(inputs);
        System.out.println("\nSmallest number is: " + inputs[output.predictedClass]);
        for(int i = 0; i < inputs.length; i++) {
            System.out.println(inputs[i] +" confidence: " + Math.round(output.outputs[i] * 10000) / 100 + "%");
        }
    }
}