package datacontrol;
import java.util.Collections;
import java.util.Arrays;
import java.util.List;

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
}
