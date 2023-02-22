package run;

import java.awt.image.*;
import java.io.File;
import java.io.FileInputStream;
import java.util.*;
import javax.imageio.ImageIO;

import datacontrol.DataHandler;
import datacontrol.DataPoint;
import datacontrol.NetworkOutput;
import neuralnetwork.NeuralNetwork;

public class Tests {
    public static void main(String[] args) {
        Mnist(true, true);
    }

    public static void Mnist(boolean useExisting, boolean train) {
        NeuralNetwork network;

        if(useExisting) {
            network = NetworkSave.LoadNetwork();
        } else {
            network = new NeuralNetwork(784, 300, 200, 50, 10);
        }

        if(train) { 
            DataPoint[] data = ReadMnist(false);

            DataHandler.Train(network, 20, data, 100, 0.8);
        }

        byte[] image = imageToBytes("run/drawing.jpg");
        double[] inputs = new double[image.length / 3];
        for(int i = 0; i < image.length; i+=3) {
            int pixel = image[i];
            if(pixel < 0) { pixel += 256; }
            inputs[i / 3] = pixel / 256.0;
        }

        NetworkOutput output = network.Classify(inputs);
        System.out.println("\nPredicted number is: " + output.predictedClass);
        for(int i = 0; i < output.outputs.length; i++) {
            System.out.println(i +" confidence: " + Math.round(output.outputs[i] * 10000) / 100 + "%");
        }
    }

    public static byte[] imageToBytes(String path) {
        try {
            WritableRaster raster = ImageIO.read(new File(path)).getRaster();
            DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
            return data.getData();
        } catch(Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static DataPoint[] ReadMnist(boolean test) {
        try {
            int size = test ? 1000 : 10000;

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

    public static void SmallerOfTwoNumbersTest(boolean useExisting) {
        Random rng = new Random();
        Scanner input = new Scanner(System.in);
        NeuralNetwork network;

        if(useExisting) {
            network = NetworkSave.LoadNetwork();
        } else {
            network = new NeuralNetwork(2, 3, 2);

            DataPoint[] randomData = new DataPoint[50000];
            for(int i = 0; i < randomData.length; i++) {
                double x = rng.nextDouble();
                double y = rng.nextDouble();
                randomData[i] = new DataPoint(new double[]{x, y}, x > y ? 1 : 0, 2);
            }

            DataHandler.Train(network, 500, randomData, 50, 0.8);

            NetworkSave.SaveNetwork(network);
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