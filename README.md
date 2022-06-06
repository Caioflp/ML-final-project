# Income Prediction on US citizens

This is the final project on the Bachelor's Machine Learning course at
EMAp-FGV. The data used was obtained [here](https://archive.ics.uci.edu/ml/datasets/Census+Income).
The objective is to predict if a given individual makes more or less than
50K USD per year.

## Report structure

* Linear regression
* Logistic Regression
* Tree methods
    - single tree
    - bagging
    - random forest
    - boosting
* Support vector machines
* Try feature selection (using variance of even the models themselves)

## Possible classification algorithms

* Linear Regression
* Logistic regression
    - Regularization
    - Optimization algorithm
    - Scale data
* K Nearest Neighbours
* Classification Trees
* SVM

## Other ideas

* Recreate missing data using something like an EM algorithm or MCMC
* Explore some unsupervised learning algorithms to find clusters in data
* Use some of the sklearn functions to perform feature reduction

## TODO

* keep or discard capital gain and capital loss features
* DO NOT use cross validated parameters for a model in another one.
  Redo the cross validation.
* Try to use categorical data with trees: one vs all type split.
  Insert a categorical variable with few labels and see what comes out.
* Discuss new methods that were used and not seen in the course
* Study over/undersampling problem

