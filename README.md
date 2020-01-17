# NaCL
Code and experiments for the paper "[What to Expect of Classifiers? Reasoning about Logistic Regression with Missing Features](http://starai.cs.ucla.edu/papers/KhosraviIJCAI19.pdf)", published in IJCAI 2019. 

Given a logistic regression (LR) model, NaCL learns a Naive Bayes model that conforms with the LR model (if used as a classifier gives out exactly same class conditional probabilities) and at the same time maximizes the joint feature likelihood P(X) on the dataset. Then, the expected prediction of the NaCL model can be used as a way of handling missing features. NaCL's learned model conforms to the original LR model, so it keeps the accuracy of the LR model when no features are missing. At the same time, it outperforms common imputation techniques in handling missing features during test time by computing expected predictions. 


# Requirements

- Python 2.7  (due to GPkit, the main library we used to solve the geometric programs).

- You can use requirements.txt to initialize a python virtual environment. Then, you can add that environment as a kernel to the ipython notebooks. The alternative is to install each requirement individually. 

- Additionally, we use "[Mosek](https://gpkit.readthedocs.io/en/latest/installation.html)" as our backend-solver which is much faster than the default solver (cvxopt). If you do not have Mosek installed simply remove `solver = 'mosek_cli'` to use cvxopt. However, we highly recommented using Mosek as its much faster and more stable.

  - There is two options to use Mosek solver in GPKIT, "mosek" or "mosek_cli". If you use "mosek_cli", GPKIT uses command line interface to call Mosek, more specifically I think it runs `mskexpopt`. More details [here](https://gpkit.readthedocs.io/en/latest/autodoc/gpkit.html). 
  - It seems GPKit might have issues with some versions of Mosek, the version used for this project is "Mosek - 8.1.0.75". For other versions, if you ran into issues trying [this](https://github.com/convexengineering/gpkit/issues/1442) might help.



# Demo
For a demo of the NaCL algorithm on how to handle missing features and its performance please refer to the [demo notebook](./notebooks/demo.ipynb).

# Experimental Results

Under the notebooks folder, we have more examples that can be used to reproduce results from the paper. For each dataset used in the paper we have a notebook that runs the experiments for that dataset and generates the results. 

Additionally, we have provided some pretrained models in the pretrained folder. 

# Runtime

Empirically, the runtime for the solver seems to grow quadratically with number of features and classes. On MNIST and Fashion datasets (with 784 features and 10 classes) the solver we used takes between 20-30min to train. On the other datasets used in the paper, NaCL took few seconds to train.

We have included implementation of NaCL using cvxpy and GPKit libraries. For best performance, we recommend using the GPKit version (the python files ending with "_GP") with Mosek backend. The ipython notebook examples use the GPKit+Mosek configuration.
