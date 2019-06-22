### we are going to try 6 models by KNN, logistic regression, naive bayes, random forest, SVM, ANN
### and in these models, the last 3 ones need to tune parameters
### so build a model with the stratified random sampling 20% data and record the tuning parameters
### to bulid a same parameter to train the 50% data and test the 50% data left.

### ANN 
  
  Number of hidden layer neurons(n) = 10, 20, ..., 100
  Epochs(ep) = 1000, 2000, ..., 10000
  Momentum constant(mc) = 0.1, 0.2, ..., 0.9
  learning rate(lr) = 0.1
### SVM
  
  Polynomials:
    Degree of kernel function(d) = 1, 2, 3, 4
    Regularization parameter(c) = 0.5, 1, 5, 10, 100
  
  Radial basis:
    Gamma in kernel function = 0.5, 1, 1.5, ..., 5, 10
    Regularization parameter(c) = 0.5, 1, 5, 10, 100

### Random Forest

  Numbers of trees = 100, 200, ..., 1000