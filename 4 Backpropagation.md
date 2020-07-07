# 4. Backpropagation

Created: Jul 6, 2020 9:26 PM

# Introduction

- backpropagation

    a way of computing gradients of expressions through recursive application of **chain rule**

- Aim

    We are given some function $f(x)$ where x is a vector of inputs and we are interested in computing the gradient of $f$  at x (i.e. $\nabla f(x)$ ).

# Simple expressions and interpretation of the gradient

- What the derivatives tell you

    They indicate the rate of change of a function with respect to that variable surrounding an infinitesimally small region near a particular point:

    $$\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}$$

    > The derivative on each variable tells you the sensitivity of the whole expression on its value.

- $\nabla f$

    the vector of partial derivatives

    $$\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]$$

    $$f(x,y) = x + y \hspace{0.1in} \rightarrow \hspace{0.1in} \frac{\partial f}{\partial x} = 1 \hspace{0.1in} \frac{\partial f}{\partial y} = 1$$

    $$f(x,y) = \max(x, y) \hspace{0.1in} \rightarrow \hspace{0.1in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.1in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)$$

# Compound expressions with chain rule

![4%20Backpropagation%2012c9d8337a1749b18272b8c65497a7d5/Untitled.png](4%20Backpropagation%2012c9d8337a1749b18272b8c65497a7d5/Untitled.png)

# Intuitive understanding of backpropagation

1. its output value and 

2. the local gradient of its output with respect to its inputs.

> This extra multiplication (for each input) due to the chain rule can turn a single and relatively useless gate into a cog in a complex circuit such as an entire neural network.

# Modularity: Sigmoid example

$$f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}$$

$$f(x) = \frac{1}{x} 
\hspace{0.1in} \rightarrow \hspace{0.1in} 
\frac{df}{dx} = -1/x^2 
\\\\
f_c(x) = c + x
\hspace{0.1in} \rightarrow \hspace{0.1in} 
\frac{df}{dx} = 1 
\\\\
f(x) = e^x
\hspace{0.1in} \rightarrow \hspace{0.1in} 
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{0.1in} \rightarrow \hspace{0.1in} 
\frac{df}{dx} = a$$

![4%20Backpropagation%2012c9d8337a1749b18272b8c65497a7d5/Untitled%201.png](4%20Backpropagation%2012c9d8337a1749b18272b8c65497a7d5/Untitled%201.png)

# Backprop in practice: Staged computation

$$f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}$$

```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```

```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

# Patterns in backward flow

![4%20Backpropagation%2012c9d8337a1749b18272b8c65497a7d5/Untitled%202.png](4%20Backpropagation%2012c9d8337a1749b18272b8c65497a7d5/Untitled%202.png)

- **add gate**

    always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass.

- **max gate**

    The **max gate:** distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass).

- **multiply gate**

    The **multiply gate**: Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule.

Notice:

- Notice that if one of the inputs to the multiply gate is very small and the other is very big, then the multiply gate will do something slightly unintuitive: it will assign a relatively huge gradient to the small input and a tiny gradient to the large input.
- Note that in linear classifiers where the weights are dot producted $w^Tx_i$ (multiplied) with the inputs, this implies that **the scale of the data has an effect on the magnitude of the gradient for the weights.**
    - For example, if you multiplied all input data examples $x_i$ by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you’d have to lower the learning rate by that factor to compensate. **This is why preprocessing matters a lot**, sometimes in subtle ways!

# Gradients for vectorized operations

**Matrix-Matrix multiply gradient**

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

[More](http://cs231n.stanford.edu/vecDerivs.pdf)