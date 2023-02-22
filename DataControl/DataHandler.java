package datacontrol;
import java.util.Collections;
import java.util.Arrays;
import java.util.List;

import neuralnetwork.NeuralNetwork;
import run.NetworkSave;

public class DataHandler {
    public static Batch[] trainingBatches;
    public static DataPoint[] testData;

    public static void splitData(DataPoint[] data, int batchSize, double split) {
        List<DataPoint> dataList = Arrays.asList(data);
        Collections.shuffle(dataList);
        Object[] objData = dataList.toArray();

        int batches = (int) (objData.length * split / batchSize);
        int testDataLength = objData.length - batchSize * batches;

        trainingBatches = new Batch[batches];
        testData = new DataPoint[testDataLength];

        for(int batch = 0; batch < batches; batch++) {
            trainingBatches[batch] = new Batch(new DataPoint[batchSize]);
            for(int i = 0; i < batchSize; i++) {
                trainingBatches[batch].data[i] = (DataPoint) objData[i + batch * batchSize];
            }
        }

        for(int i = 0; i < data.length - batches * batchSize; i++) {
            testData[i] = (DataPoint) objData[i + batches * batchSize];
        }
    }

    public static void Train(NeuralNetwork network, int epochs, DataPoint[] data, int batchSize, double split) {
        for(int epoch = 0; epoch < epochs; epoch++) {
            DataHandler.splitData(data, batchSize, split);

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

            System.out.println("Epoch: " + (epoch + 1) + " Test accuracy: " + 100.0 * correct / DataHandler.testData.length + "%");

            NetworkSave.SaveNetwork(network);
        }
    }
}
