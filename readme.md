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

<img alt="$loss = classification \ loss + \sum_i  \frac{1}{2}\lambda\mu^2 - \log\sigma_i + \frac{1}{2} \lambda\sigma^2 + constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/4bec59300983a8f13a76de9aff3699d0.svg?invert_in_darkmode" align=middle width="468.854595pt" height="27.720329999999983pt"/>

Actually, our new loss function seems remarkably similar to the old loss function, except for the <img alt="$\frac{1}{2}\lambda\sigma^2 - \log \sigma$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/5d911faa1558935a5941410cdd0c0c38.svg?invert_in_darkmode" align=middle width="89.36234999999999pt" height="27.720329999999983pt"/>. One could consider that the *loss function* for the standard deviations.

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

I made a separate document in [/docs/](https://github.com/RobRomijnders/weight_uncertainty/blob/master/docs/explaining_var_inf_fact_norm.pdf) to explain in a formal sense why this new loss function works for approximation the parameter posterior. Please read it at your own risk :) You can read, use and enjoy this entire repository without ever reading it. 

# Making predictions with uncertainty
Now that we have a posterior, let's use it to make predictions and get uncertainties. What we want to know is the probability for an output class, given the input. We would get that like so

<img alt="$p(y|x) = \int_w p(p|x,w)p(w|data)dw$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/dbb647613f36e527b72e9cdccbd6c3ce.svg?invert_in_darkmode" align=middle width="239.23399499999996pt" height="26.48447999999999pt"/>

It would take forever to integrate over all possible parameters. (Remember that neural networks have millions of parameters, so computing that integral requires integrating over millions of real dimensions). Fortunately, we can easily sample parameters and use it to approximate the integral. Again, we love python at PyData, so let's write some python:

```python
def sample_prediction(input):
    for _ in range(num_samples):
        w = approximation.sample()  
        yield model.predict(input, w)
prediction = np.mean(sample_prediction(input))

``` 
??? TODO ??? add corresponding code reference

What does this code do?

  * For many times, we sample a parameter vector from our approximation. We use the sampled parameter vector to make a prediction
  * The approximation of our predictive distribution <img alt="$p(y|x)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/fc76db86ea6c427fdd05067ba4835daa.svg?invert_in_darkmode" align=middle width="43.50555pt" height="24.56552999999997pt"/> is the average of all the sampled predictions. 

In this project, we work with classification. Therefore, <img alt="$p(y|x)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/fc76db86ea6c427fdd05067ba4835daa.svg?invert_in_darkmode" align=middle width="43.50555pt" height="24.56552999999997pt"/> is a vector of `num_classes` dimension. Each entry in the vector tells the probability that the input belongs to that class.

For example, if our classification problem concerns cats, dogs and cows. Then `prediction[1]` tells the probability that in input is a dog.

### Getting the uncertainty

How do we get **one** number that tells us the uncertainty of our prediction? We have a full posterior predictive distribution, <img alt="$p(y|x)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/fc76db86ea6c427fdd05067ba4835daa.svg?invert_in_darkmode" align=middle width="43.50555pt" height="24.56552999999997pt"/>. We want one number that quantifies the uncertainty. 

There are many choices for this one number to summarize the uncertainty

  * Use the predicted probability `prediction[i]`
  * Use the variance in the predicted probabilities `np.var(sample_prediction(input))[i]`
  * Use the variation ratio `np.mean(np.argmax(sample_prediction(input),axis=1))`
  * Use the predictive entropy `entropy(prediction)`
  * Use the mutual information between parameters and labels `entropy(prediction) - np.mean(entropy(sample_predictions(input),axis=1))`

If you are interested in comparing these uncertainty quantifiers, [this paper](https://arxiv.org/pdf/1803.08533.pdf) compares them. 

What we really care about is which uncertainty quantifier makes us robust againt adversarial attacks. Fortunately, the authors of [this paper](https://arxiv.org/pdf/1711.08244.pdf) compare the uncertainty quantifiers when under adversarial attacks. They conclude that both the variation ratio, predictive entropy and the mutual information increase for adversarial inputs. I care about simplicity, so I will use the predictive entropy in the rest of the project.

# How to prune the parameters?
Now let's answer how to prune the parameters. We have neural network with millions of weights. We want to drop many of them or at least zero them out. The question we face is the following: *which parameters should we drop first?*. 

Intuitively, we drop the parameters first that is least useful. For example, if a parameter has a high posterior probability of being zero, we might as well drop it. Conversely, if a parameter has a low posterior probability of being zero, we want to keep it. We follow this intuition as we prune parameters: 1) we pick a threshold for the zero probability and 2) we sweep over all the parameters and drop the ones whose probability at zero is above the threshold. 

Again, PyData loves python, so let's write some python

```python
for param, mu, sigma in approximation():
    zero_probability = normal.pdf(mu, sigma, 0.0)
    if zero_probability > threshold:
        model.drop(param)
```
For the corresponding code in the project, see: `utils/model.Model.pruning()`

# Experiments and results
For the experiments, we run the Bayesian neural network on three data set:

  * First, we want an easy data set that everyone understands. Therefore, we pick MNIST
  * Second, we want an application that many people care about: image classification. Therefore, we pick [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html). It is also more applicable than MNIST
  * Third, we want a time series data set, as it is a common application of neural networks. We also want to show that Bayesian neural networks do not overfit. Therefore, we pick the ECG5000 data set from [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/). The train set contains only 500 time series, so we know that a conventional neural network would overfit.

For each data set, we care about the following experiments

  * How does the pruning curve look like? Do we remain performance as we drop the parameters?
  * What do examples of certain and uncertain inputs look like? Does uncertainty increase for noisy inputs?

To this end, we have three plots per data set:

  * __A pruning curve__: the horizontal axis changes the portion of weights being dropped. The vertical axis indicates the validation performance. We expect that the validation performance remains good when less than 90% of the parameters are dropped. (That is also the title of the PyData talk)
  * __Examples of inputs__: we randomly sample some images from the validation set and we mutilate them by either adding noise or rotating them. As mutilation increases, we expect the uncertainty to increase too.
  * __Uncertainty curves__: we dive further in our uncertainty numbers and our expectation that they increase for more mutilation. For each mutilation, we plot the uncertainty number as a function of mutilation value (like the energy of the noise or the angle of rotation). This plot will confirm on aggregate level that uncertainty increases for more mutilation.

### MNIST
Pruning curve
![pruning_curve_mnist](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/pruning_curves/mnist_pruning_curve.png?raw=true)

Examples with noise
![mnist_noise](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/mnist/noise/noise_uncertain.gif?raw=true)

Examples with rotation
![mnist_rotation](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/mnist/rotation/rotation_uncertain.gif?raw=true)

Uncertainty curve
![mnist_uncertain_curve](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/uncertainty_curves/mnist_uncertainty_curve.png?raw=true)

### CIFAR10

### MNIST
Pruning curve
![pruning_curve_cifar](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/pruning_curves/cifar_pruning_curve.png?raw=true)

Examples with noise
![cifar_noise](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/cifar/noise/noise_uncertain.gif?raw=true)

Examples with rotation
![cifar_rotation](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/cifar/rotation/rotation_uncertain.gif?raw=true)

Uncertainty curve
![cifar_uncertain_curve](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/uncertainty_curves/cifar_uncertainty_curve.png?raw=true)

### ECG5000


# Further reading

  * The idea was recently described in [Weight Uncertainty in Neural Networks (2015)](https://arxiv.org/abs/1505.05424)
    * The paper builds on an older paper [Practical Variational Inference for Neural Networks(2011)](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)
      * and that paper builds on an even older paper [Keeping the neural networks simple by minimizing the description length of the weights (1993)](https://dl.acm.org/citation.cfm?id=168306)
  * Also this paper gives another view variational inference for neural networks [Bayesian Compression for Deep Learning (2017)](https://arxiv.org/abs/1705.08665)
  * Speed comparisons of pruned neural networks [Here](https://arxiv.org/abs/1705.08665)
  * Corner stone thesis on [Bayesian neural networks by Radford Neal (1995)](https://www.cs.toronto.edu/~radford/ftp/thesis.ps)
  * For more fundamental reading on variational inference:
    * Chapter 21 on Variational Inference of [Machine learning: a probabilistic perspective](https://www.cs.ubc.ca/~murphyk/MLbook/)
    * Chapter 10 on Approximate Inference of [Pattern recognition and machine learning](https://www.springer.com/gp/book/9780387310732)
    * Chapter 33 on Variational Methods of [Information theory, inference and learning algorithms](http://www.inference.org.uk/itila/book.html)
  * [This paper](https://arxiv.org/pdf/1703.02910v1.pdf) uses Bayesian Neural networks for active learning. They show that querying with the uncertainty from a Bayesian Neural Network reaches higher performance faster.
  * Two papers discuss the uncertainty quantifiers for Bayesian neural networks
    * [This paper](https://arxiv.org/pdf/1803.08533.pdf) describes the differences between the uncertainty quantifiers
    * [This paper](https://arxiv.org/pdf/1711.08244.pdf) compares the uncertainty quantifiers for adversarial and out-of-distribution inputs
