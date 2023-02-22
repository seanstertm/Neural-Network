package run;

import activation.*;
import loss.*;

public class CONFIG {
    public static Activation activationType = new ReLU();
    public static Loss lossType = new SquaredResiduals();

    public static double learnRate = 6;
    public static double regularization = 0.15;
    public static double momentum = 0.9;
}