
# Note

This branch is WIP refactoring of moving to python 3. The old environment I used does not work anymore and not supported by GPKit. Some experiment code may be removed for official code for the paper checkout the master branch. However, this should be easier to use.

Main code is now under `./src`. The examples in the `./examples` folder have been ported to new code. Some functions under `src/util.py` might not have been ported yet. In the `./notebooks` folder most notebooks are ported and should work now, except the notebook for the explanations that one is still not ported.


# NaCL: Naive Conformant Learning
Code the paper "[What to Expect of Classifiers? Reasoning about Logistic Regression with Missing Features](http://starai.cs.ucla.edu/papers/KhosraviIJCAI19.pdf)", published in IJCAI 2019.

Given a logistic regression (LR) model, NaCL learns a Naive Bayes model that conforms with the LR model (if used as a classifier gives out exactly same class conditional probabilities) and at the same time maximizes the joint feature likelihood P(X) on the dataset. Then, the expected prediction of the NaCL model can be used as a way of handling missing features. NaCL's learned model conforms to the original LR model, so it keeps the accuracy of the LR model when no features are missing. At the same time, it outperforms common imputation techniques in handling missing features during test time by computing expected predictions.

# Requirements

- Python3, GPkit v0.9.9

- You can use requirements.txt to initialize a python virtual environment. Then, you can add that environment as a kernel to the ipython notebooks. The alternative is to install each requirement individually.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt --user
```

- Additionally, we use "[Mosek 9.2.28](https://gpkit.readthedocs.io/en/latest/installation.html)" as our backend-solver which is much faster than the default solver (cvxopt). If you do not have Mosek installed simply remove the default is to use cvxopt. However, we highly recommented using Mosek as its much faster and more stable. You can do that by setting `solver = 'mosek_cli'` or `solver = 'mosek_conif'`. 

The following script installs the mentioned MOSEK9 in your home directory. (for `mosek_conif` you need mosek 9)
```bash
wget https://download.mosek.com/stable/9.2.28/mosektoolslinux64x86.tar.bz2
tar -xvf mosektoolslinux64x86.tar.bz2 -C ~
export PATH=$PATH:$HOME/mosek/9.2/tools/platform/linux64x86/bin
```

Or if you want MOSEK 8 you can. (for `mosek_cli` you need mosek8)
```bash
wget https://download.mosek.com/stable/8.1.0.82/mosektoolslinux64x86.tar.bz2
tar -xvf mosektoolslinux64x86.tar.bz2 -C ~
export PATH=$PATH:$HOME/mosek/8/tools/platform/linux64x86/bin/
```


Also need to request a license from Mosek (its free for academic usecases), and copy it to `~/mosek/mosek.lic`. Also consider adding the `export PATH...` into your `.bashrc`.


Note: If you install Mosek after you install GPkit you need might need to uninstall and install gpkit again.
Make sure to double check latest instruction at GPKit and Mosek.

# Examples
You can run the examples as follows (here --remove is used to remove some pixels around the picture to make optimization go faster and for benchmarking purposes. If you want all pixels don't add the argument, for mnist or fashion it should take about 20-30min and you might run out of memory). The main bottleneck is number of features. Having more data it should scale linearly but scales quadratically for more features, classes.

```bash
python3 examples/fashion.py --remove 8

python3 examples/mnist.py --remove 8

python3 examples/splice.py
```

# Notebook Demos
For a demo of the NaCL algorithm on how to handle missing features and its performance please refer to the [demo notebook](./notebooks/demo.ipynb).

# Experimental Results

Under the notebooks folder, we have more examples that can be used to reproduce results from the paper. For each dataset used in the paper we have a notebook that runs the experiments for that dataset and generates the results. 

Additionally, we have provided some pretrained models in the pretrained folder. 

# Runtime

Empirically, the runtime for the solver seems to grow quadratically with number of features and classes. On MNIST and Fashion datasets (with 784 features and 10 classes) the solver we used takes between 20-30min to train. On the other datasets used in the paper, NaCL took few seconds to train.
