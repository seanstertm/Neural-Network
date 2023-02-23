package datacontrol;

// Class that contains both the output of the network and
// what the network thinks the correct answer is
public class NetworkOutput {
    public int predictedClass;
    public double[] outputs;

    public NetworkOutput(double[] outputs, int predictedClass) {
        this.outputs = outputs;
        this.predictedClass = predictedClass;
    }

    // Ease of use function to print the outputs
    public String toString() {
        String sentence = "returned index: " + predictedClass;
        for(double output : outputs) {
            sentence += "\n" + output;
        }
        return sentence;
    }
}