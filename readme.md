This repository is under heavy development. Please use at own risk. I plan to finish it before PyData on May 25th.

# Introduction
This repository implements Bayesian inference on a deep neural network. The repository also serves as notes for my talk at PyData Amsterdam 2018 **Bayesian Deep Learning with 10 % of the weights** 

# Motivation
Conventional neural networks suffer from two problems

  * Conventional neural networks give no **uncertainty** on their predictions. 
    * This is detrimental for critical applications. For example, if a neural network diagnoses you with a disease, wouldn't you want to know how certain it is of that diagnosis?
    * This also makes neural networks susceptible to adversarial attacks. In adversarial attacks, imperceptible changes to the input results in vastly different predictions. Bayesian deep learning will not combat adversarial attacks, but it will show increased uncertainty on adversarial attacks.

  * Conventional neural networks have **millions of parameters**

    * This is detrimental for mobile applications. In mobile applications, we often have small memory and not much computation power. If we can prune the parameters, we would take up less memory and need fewer compute to make a prediction
    * (There are some speculations that the redundant parameters make it easier for adversarial attacks, but that is just a hypothesis.)

Fortunately, this repository proposes a solution to both problems in one simple method: Bayesian inference

In Bayesian inference, we infer a posterior over our parameters <img alt="$p(w|data) \propto p(data|w)p(w)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ae6c870a717604621ad875abaa1b936d.svg?invert_in_darkmode" align=middle width="193.903545pt" height="24.56552999999997pt"/>.  This posterior helps us in two ways:
  
  * The predictions using the parameter posterior naturally give us uncertainty in our predictions. <img alt="$p(y|x) = \int_w p(p|x,w)p(w|data)dw$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/dbb647613f36e527b72e9cdccbd6c3ce.svg?invert_in_darkmode" align=middle width="239.23399499999996pt" height="26.48447999999999pt"/>
  * The posterior tells us which parameters assign a high probability to being zero. We will prune these parameters.


# Approximation of parameter posterior

Before we talk about approximations, let us first write down the posterior. For the posterior, we need a likelihood and a prior. In this repository we deal with classification, so our likelihood is the probability of the prediction for the correct class. We choose a Gaussian prior over our parameters. The prior might sound like a new concept to many people, but I want to convince you that we have been using priors all the time. When we do *L2 regularisation* or when we do *weight decay*, that corresponds to assuming a Gaussian prior on the parameters.

<img alt="$p(w|data) \propto p(data|w)p(w)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ae6c870a717604621ad875abaa1b936d.svg?invert_in_darkmode" align=middle width="193.903545pt" height="24.56552999999997pt"/>

<img alt="$log p(w|data) =  log p(data|w) + log p(w) + constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/e31c20782efe67b634accb5aabf87054.svg?invert_in_darkmode" align=middle width="361.48579499999994pt" height="24.56552999999997pt"/>

<img alt="$log p(w|data) =  classification \ loss + \lambda \sum_i w_i^2+ constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ffa76ab98ff1da7adf2574b4fba0caac.svg?invert_in_darkmode" align=middle width="409.62454499999996pt" height="26.70657pt"/>

Now we want to approximate this posterior to use it for uncertain predictions and parameter pruning. Actually, we have been approximating this posterior all our lives. We train neural networks with Stochastic gradient descent to find the best parameters. This corresponds to approximating our posterior with one paramers vector. *Formal people would say that we make a point approximation: one point for the optimal parameter vector*. 

In our case, we want a richer approximation than a point approximation. But we also do not want to overcomplicate matters. Therefore, we approximate the posterior with a Gaussian. The Gaussian is ideal, because:

  * The Gaussian distribution can capture the local properties of the true posterior. This will get us the uncertainty in our predictions
  * The Gaussian distribution has a simple form that we can use for pruning. Each parameter will have a mean and a standard deviation. With the mean and standard deviation, we calculate the zero probability and prune accordingly.

## Loss function

We will first discuss how we find the best approximation for our posterior. Later, we will show a bit of the formal math.

Remember that the loss function for the true posterior was:

<img alt="$log p(w|data) =  classification \ loss + \lambda \sum_i w_i^2+ constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ffa76ab98ff1da7adf2574b4fba0caac.svg?invert_in_darkmode" align=middle width="409.62454499999996pt" height="26.70657pt"/>

Now let us name our approximation <img alt="$q(w)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/68d98207e0a01db0b474b8cf789ab914.svg?invert_in_darkmode" align=middle width="32.8053pt" height="24.56552999999997pt"/>, then our new loss function is:

<img alt="$log p(w|data) =  classification \ loss + \lambda \sum_i (\frac{\mu_i}{\sigma_i})^2 + log(\sigma_i) +\sigma_i^{-2} + constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/13033632a4da3583256dd22e1b6781c1.svg?invert_in_darkmode" align=middle width="541.401795pt" height="28.839689999999997pt"/>

Actually, our new loss function seems remarkably similar to the old loss function, except for the <img alt="$log(\sigma_i) +\sigma_i^{-2}$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/3c9cfadb06e9a5c0ae3c32f6b19595d3.svg?invert_in_darkmode" align=middle width="95.872755pt" height="28.839689999999997pt"/>. One could consider that the *loss function* for the standard deviations.

At PyData, we love python. So let's write this out in python.

We would train conventional neural networks like so:

```python
while not converged:
  # Get the loss
  x, y = sample_batch()
  loss = loss_function(x, y, w)

  #Update the parameters
  w_grad = gradient(loss, w)
  w = update(w, w_grad)
```
In Bayesian inference, we make an approximation to the posterior. So we would approximate the posterior like so

```python
while not converged:
  # Get the loss
  x, y = sample_batch()
  w = approximation.sample()
  loss = loss_function(x, y, w)

  # Update the approximation
  w_grad = gradient(loss, w)
  approximation = update(approximation, w_grad)
```

I made a separate document to explain in a formal sense why this new loss function works for approximation the parameter posterior. Please read it at your own risk :) You can read, use and enjoy this entire repository without ever reading it. 



# Pruning

# Experiments and results



# Further reading

  * The idea was recently described in [Weight Uncertainty in Neural Networks (2015)](https://arxiv.org/abs/1505.05424)
    * The paper builds on an older paper [Practical Variational Inference for Neural Networks(2011)](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)
      * and that paper builds on an even older paper [Keeping the neural networks simple by minimizing the description length of the weights (1993)](https://dl.acm.org/citation.cfm?id=168306)
  * Also this paper gives another view variational inference for neural networks [Bayesian Compression for Deep Learning (2017)](https://arxiv.org/abs/1705.08665)
  * For more fundamental reading on variational inference:
    * Chapter 21 on Variational Inference of [Machine learning: a probabilistic perspective](https://www.cs.ubc.ca/~murphyk/MLbook/)
    * Chapter 10 on Approximate Inference of [Pattern recognition and machine learning](https://www.springer.com/gp/book/9780387310732)
    * Chapter 33 on Variational Methods of [Information theory, inference and learning algorithms](http://www.inference.org.uk/itila/book.html)
