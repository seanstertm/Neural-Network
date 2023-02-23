package run;

import activation.*;
import loss.*;

// Essentially a static class to save configuration data
// For Mnist, activation type is ReLU
public class CONFIG {
    public static Activation activationType = new ReLU();
    public static Loss lossType = new SquaredResiduals();

    public static double learnRate = 0.5;
    public static double regularization = 0.15;
    public static double momentum = 0.9;
}