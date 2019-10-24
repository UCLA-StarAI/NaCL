# NaCL
Code and experiments for the paper "[What to Expect of Classifiers? Reasoning about Logistic Regression with Missing Features](http://starai.cs.ucla.edu/papers/KhosraviIJCAI19.pdf)", published in IJCAI 2019. 

Given a logistic regression (LR) model, NaCL learns a Naive Bayes model that conforms with the LR model (if used as a classifier gives out exactly same class conditional probabilities) and at the same time maximizes the joint feature likelihood P(X) on the dataset. Then, the Naive Bayes model can be used to handle missing features while having performance as good as LR when no features are missing.


# Requirements

You can use requirements.txt to initialize a python virtual environment. Then, you can add that environment as a kernel to the ipython notebooks. The alternative is to install each requirement individually. 

The GPKit library we use to solve the geometric programs requires python 2 version.

# Usage

For examples of how to run the code, please refer to the notebooks folder. Additionally, we have provided some pretrained models in the pretrained folders. You can use those to repeat the experiments in the paper.

# Runtime

Empirically, the runtime for the solver seems to grow quadratically with number of features and classes. On MNIST and Fashion datasets (with 784 features and 10 classes) the solver we used takes between 20-30min to train.

We included implementation of NaCL in two Geometric Programming library. For best performance, we recommend using the GPKit version (the python files ending with "_GP") with Mosek backend. The ipython notebook examples use the GPKit version.
