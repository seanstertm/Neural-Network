package datacontrol;
public class NetworkOutput {
    public int predictedClass;
    public double[] outputs;

    public NetworkOutput(double[] outputs, int predictedClass) {
        this.outputs = outputs;
        this.predictedClass = predictedClass;
    }

    public String toString() {
        String sentence = "returned index: " + predictedClass;
        for(double output : outputs) {
            sentence += "\n" + output;
        }
        return sentence;
    }
}