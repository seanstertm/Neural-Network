package DataControl;
public class DataPoint {
    public double[] inputs;
    public double[] expectedOutputs;
    public int label;

    public DataPoint(double[] inputs, int label, int numLabels) {
        this.inputs = inputs;
        this.label = label;
        this.expectedOutputs = GenerateExpectedOutputs(label, numLabels);
    }

    private static double[] GenerateExpectedOutputs(int index, int size) {
        double[] output = new double[size];
        output[index] = 1;
        return output;
    }
}
