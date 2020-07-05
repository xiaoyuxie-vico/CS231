# 3. Optimization

Created: Jul 4, 2020 6:38 PM

# Introduction

Three components:

- score function

    A (parameterized) score function mapping the raw image pixels to class scores (e.g. a linear function)

- loss function

    A loss function that measured the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data. We saw that there are many ways and versions of this (e.g. Softmax/SVM).

- optimization

    Optimization is the process of finding the set of parameters $W$ that minimize the loss function.

Further reading:

- [convex optimization](http://stanford.edu/~boyd/cvxbook/)

# Optimization

To reiterate, the loss function lets us quantify the quality of any particular set of weights W. The goal of optimization is to find W that minimizes the loss function.

keep in mind that our goal is to eventually optimize Neural Networks where we can’t easily use any of the tools developed in the Convex Optimization literature.

## Strategy #1: A first very bad idea solution: Random search

```python
# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (trunctated: continues for 1000 lines)
```

```python
# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555
```

### Core idea: iterative refinement

> Our strategy will be to start with random weights and iteratively refine them over time to get lower loss

### Blindfolded hiker analogy

One analogy that you may find helpful going forward is to think of yourself as hiking on a hilly terrain with a blindfold on, and trying to reach the bottom.

acc: 15.5%

## Strategy #2: Random Local Search

The first strategy you may think of is to try to extend one foot in a random direction and then take a step only if it leads downhill.

Concretely, we will start out with a random $W$, generate random perturbations $\delta W$ to it and if the loss at the perturbed $W + \delta W$ is lower, we will perform an update.

```python
W = np.random.randn(10, 3073) * 0.001 # generate random starting W
bestloss = float("inf")
for i in range(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  print 'iter %d loss is %f' % (i, bestloss)
```

acc: 21.4%.

## Strategy #3: Following the Gradient

In the previous section we tried to find a direction in the weight-space that would improve our weight vector (and give us a lower loss). It turns out that there is no need to randomly search for a good direction: **we can compute the best direction along which we should change our weight vector that is mathematically guaranteed to be the direction of the steepest descend (at least in the limit as the step size goes towards zero)**. This direction will be related to the gradient of the loss function. In our hiking analogy, this approach roughly corresponds to feeling the slope of the hill below our feet and stepping down the direction that feels steepest.

**Derivatives**

$$\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}$$

When the functions of interest take a vector of numbers instead of a single number, we call the derivatives **partial derivatives**, and the gradient is simply the vector of partial derivatives in each dimension.

# Computing the gradient

- numerical gradient
- analytic gradient

## Computing the gradient numerically with finite differences

```python
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```

**Practical considerations**

in practice it often works better to compute the numeric gradient using the centered difference formula: $[f(x+h)−f(x−h)]/2h$ . See [wiki](http://en.wikipedia.org/wiki/Numerical_differentiation) for details.

```python
# to use the generic code above we want a function that takes a single argument
# (the weights in our case) so we close over X_train and Y_train
def CIFAR10_loss_fun(W):
  return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001 # random weight vector
df = eval_numerical_gradient(CIFAR10_loss_fun, W) # get the gradient
```

```python
loss_original = CIFAR10_loss_fun(W) # the original loss
print 'original loss: %f' % (loss_original, )

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
  step_size = 10 ** step_size_log
  W_new = W - step_size * df # new position in the weight space
  loss_new = CIFAR10_loss_fun(W_new)
  print 'for step size %f new loss: %f' % (step_size, loss_new)

# prints:
# original loss: 2.200718
# for step size 1.000000e-10 new loss: 2.200652
# for step size 1.000000e-09 new loss: 2.200057
# for step size 1.000000e-08 new loss: 2.194116
# for step size 1.000000e-07 new loss: 2.135493
# for step size 1.000000e-06 new loss: 1.647802
# for step size 1.000000e-05 new loss: 2.844355
# for step size 1.000000e-04 new loss: 25.558142
# for step size 1.000000e-03 new loss: 254.086573
# for step size 1.000000e-02 new loss: 2539.370888
# for step size 1.000000e-01 new loss: 25392.214036
```

**Update in negative gradient direction**

**Effect of step size**

![3%20Optimization%2073c64885cf684988ab412d56343afa36/Untitled.png](3%20Optimization%2073c64885cf684988ab412d56343afa36/Untitled.png)

Visualizing the effect of step size.

## Computing the gradient analytically with Calculus

**Gradient check**: compute the analytic gradient and compare it to the numerical gradient to check the correctness of your implementation.

$$L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]$$

$$\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i$$

where $\mathbb{1}$ is the indicator function that is one if the condition inside is true or zero otherwise.

For the other rows where $j≠y_i$ the gradient is:

$$\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i$$

### Gradient Descent

**Vanilla Gradient Descent:**

```python
# Vanilla Gradient Descent

while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
```

**Mini-batch gradient descent.**

Reason:

- It seems wasteful to compute the full loss function over the entire training set in order to perform only a single parameter update.
- the examples in the training data are **correlated**.

    To see this, consider the extreme case where all 1.2 million images in ILSVRC are in fact made up of exact duplicates of only 1000 unique images (one for each class, or in other words 1200 identical copies of each image). Then it is clear that the gradients we would compute for all 1200 identical copies would all be the same, and **when we average the data loss over all 1.2 million images we would get the exact same loss as if we only evaluated on a small subset of 1000**. In practice of course, the dataset would not contain duplicate images, **the gradient from a mini-batch is a good approximation of the gradient of the full objective**. Therefore, much **faster convergence** can be achieved in practice by evaluating the mini-batch gradients to perform more frequent parameter updates.

Example:

- a typical batch contains 256 examples from the entire training set of 1.2 million.

```python
# Vanilla Minibatch Gradient Descent

while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```

**Stochastic Gradient Descent (SGD) (**or also sometimes **on-line** gradient descent)

a setting where the mini-batch contains only a single example

- **less common to see**

    in practice due to **vectorized code optimizations** it can be computationally much more efficient to evaluate the gradient for 100 examples, than the gradient for one example 100 times.

- Other name

    use the term SGD even when referring to mini-batch gradient descent (i.e. mentions of MGD for “Minibatch Gradient Descent”, or BGD for “Batch gradient descent” are rare to see)

- The size of the mini-batch

    The size of the mini-batch is a hyperparameter but it is not very common to cross-validate it. It is usually based on memory constraints (if any), or set to some value, e.g. 32, 64 or 128. We use powers of 2 in practice because many vectorized operation implementations work faster when their inputs are sized in powers of 2.

# Summary

![3%20Optimization%2073c64885cf684988ab412d56343afa36/Untitled%201.png](3%20Optimization%2073c64885cf684988ab412d56343afa36/Untitled%201.png)

Keywords:

- high-dimensional optimization landscape
- iterative refinement
- gradient
- step size (or the learning rate)
- numerical and analytic gradient
- gradient check
- Gradient Descent