package datacontrol;

// Class to hold data and its expected outputs
public class DataPoint {
    public double[] inputs;
    public double[] expectedOutputs;
    public int label;

    public DataPoint(double[] inputs, int label, int numLabels) {
        this.inputs = inputs;
        this.label = label;
        this.expectedOutputs = GenerateExpectedOutputs(label, numLabels);
    }

    // Returns an array like [0, 1, 0, 0] based on the index of the correct answer
    private static double[] GenerateExpectedOutputs(int index, int size) {
        double[] output = new double[size];
        output[index] = 1;
        return output;
    }
}
