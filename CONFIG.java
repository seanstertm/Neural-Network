public class CONFIG {
    public static Activation activationType = new Sigmoid();
    public static Loss lossType = new SquaredResiduals();

    public static double learnRate = 1;
    public static double regularization = 0.15;
    public static double momentum = 0.9;
}
