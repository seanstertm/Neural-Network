package neuralnetwork;

// This class holds the data for each layer to later be applied as opposed to using the same layer class
public class LayerLearnData {
	public double[] inputs;
	public double[] weightedInputs;
	public double[] activations;
	public double[] nodeValues;

	public LayerLearnData(Layer layer)
	{
		weightedInputs = new double[layer.numNodesOut];
		activations = new double[layer.numNodesOut];
		nodeValues = new double[layer.numNodesOut];
	}
}