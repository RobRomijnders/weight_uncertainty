# Introduction
The code in this repository implements Bayesian inference on a deep neural network. The repository also serves as notes for my talk at PyData Amsterdam 2018 **Bayesian Deep Learning with 10 % of the weights** 

# Getting started
Move your console to the outer `weight_uncertainty` directory:
> cd weight_uncertainty   (you are now in the outer weight_uncertainty directory)
> pip install -e .
> python weight_uncertainty/main.py

These commands install the repo and run the training process

# Motivation
Conventional neural networks suffer from two problems, which motivate this repository:

  * Conventional neural networks give no **uncertainty** on their predictions. 
    * This is detrimental for critical applications. For example, if a neural network diagnoses you with a disease, wouldn't you want to know how certain it is of that diagnosis?
    * This also makes neural networks susceptible to adversarial attacks. In adversarial attacks, imperceptible changes to the input results in vastly different predictions. We desire that a neural network gives high uncertainty when we input an adversarial input.

  * Conventional neural networks have **millions of parameters**

    * This is detrimental for mobile applications. In mobile applications, we often have small memory and not much computation power. If we can prune the parameters, we would take up less memory and need fewer compute to make a prediction
    * (There are some speculations that the redundant parameters make it easier for adversarial attacks, but that is just a hypothesis.)

This repository proposes a solution to both problems.

## Short summary of solution
In short: in conventional learning of neural nets, we use SGD to find one parameter vector. In this project, we are going to find multiple parameter vectors. When making a prediction, we average the outputs of the neural net with each parameter vector. You can think of this as an ensemble method. 

I hear you asking: how do we get multiple parameter vectors? Answer: we sample them from the posterior over our parameters.

We infer a posterior over our parameters according to Bayes rule: <img alt="$p(w|data) \propto p(data|w)p(w)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ae6c870a717604621ad875abaa1b936d.svg?invert_in_darkmode" align=middle width="193.903545pt" height="24.56552999999997pt"/>.  This posterior helps us in two ways:
  
  * The predictions using the parameter posterior naturally give us uncertainty in our predictions. <img alt="$p(y|x) = \int_w p(p|x,w)p(w|data)dw$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/dbb647613f36e527b72e9cdccbd6c3ce.svg?invert_in_darkmode" align=middle width="239.23399499999996pt" height="26.48447999999999pt"/>
  * The posterior tells us which parameters assign a high probability to being zero. We will prune these parameters.


# Parameter posterior

Let us first write down the posterior. For the posterior, we need a likelihood and a prior. In this repository we deal with classification, so our _likelihood_ is the probability of the prediction for the correct class. We choose a Gaussian _prior_ over our parameters. The prior might sound like a new concept to many people, but I want to convince you that we have been using priors all the time. When we do *L2 regularisation* or when we do *weight decay*, that corresponds to assuming a Gaussian prior on the parameters.

<img alt="$p(w|data) \propto p(data|w)p(w)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ae6c870a717604621ad875abaa1b936d.svg?invert_in_darkmode" align=middle width="193.903545pt" height="24.56552999999997pt"/>

<img alt="$log p(w|data) =  log p(data|w) + log p(w) + constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/e31c20782efe67b634accb5aabf87054.svg?invert_in_darkmode" align=middle width="361.48579499999994pt" height="24.56552999999997pt"/>

<img alt="$log p(w|data) =  classification \ loss + \lambda \sum_i w_i^2+ constant$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ffa76ab98ff1da7adf2574b4fba0caac.svg?invert_in_darkmode" align=middle width="409.62454499999996pt" height="26.70657pt"/>

So actually, we have been using the parameter posterior all the time when we did L2 regularisation. However, in conventional learning, we used only one parameter vector from this posterior. In this repository, we want to _sample_ multiple parameter vectors from the posterior.

## How do we sample from the posterior?
Exact sampling from the posterior is hard. Therefore, we make a local approximation to the posterior that we can easily sample. We want a richer approximation than a point approximation. But we also do not want to overcomplicate matters. Therefore, we approximate the posterior with a Gaussian. The Gaussian is ideal, because:

  * The Gaussian distribution can capture the local structure of the true posterior. This will tell us about the behavior of parameter vectors: which parameters can assume a wide range of values, and which parameters are fairly restricted.
  * The Gaussian distribution has a simple form that we can use for pruning. Each parameter will have a mean and a standard deviation. With the mean and standard deviation, we calculate the zero probability in one simple line. So pruning will be efficient.

## Loss function

We will find our approximation via stochastic gradient descent. This time, however, the loss function for SGD differs a little bit.

Remember that the old loss function was:

<img alt="$log p(w|data) =  classification \ loss + \lambda \sum_i w_i^2$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/38f09888360020352fcd6ca86088359a.svg?invert_in_darkmode" align=middle width="325.88704499999994pt" height="26.70657pt"/>

Then our new loss function becomes:

<img alt="$loss = classification loss + \sum_i - \log\sigma_i + \frac{1}{2}\lambda \sigma^2 +  \frac{1}{2}\lambda\mu^2$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/ce7ceec6190c3e271d9cc08bc56b3359.svg?invert_in_darkmode" align=middle width="395.116095pt" height="27.720329999999983pt"/>

### What changed in the loss function?

  * Both loss functions have the classification loss
  * Both loss functions have a squared penalty on the mean of the parameter vector
  * The new loss function has an additional penalty on <img alt="$\sigma$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode" align=middle width="9.945705000000002pt" height="14.102549999999994pt"/>. This penalty _penalizes_ small sigma's. In other words, this loss function _promotes_ large values of sigma. In the `im` directory, you find a figure of this penalty term, named `loss_sigma.png`

### Let's see some code

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
Now that we have sampled parameter vectors, let's use them to make predictions and get uncertainties. What we want to know is the probability for an output class, given the input. We will make this prediction by averaging the output of the neural net with each of the parameter vectors:

Again, we love python, so let's write some python:

```python
def sample_prediction(input):
    for _ in range(num_samples):
        w = approximation.sample()  
        yield model.predict(input, w)
prediction = np.mean(sample_prediction(input))

``` 
(`RestoredModel.predict()` in `util.util.py` implements exactly this)

What does this code do?

  * For many times, we sample a parameter vector from our approximation. We use the sampled parameter vector to make one prediction
  * Our final prediction is the average of all the sampled predictions. 

In this project, we work with classification. Therefore, <img alt="$p(y|x)$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/fc76db86ea6c427fdd05067ba4835daa.svg?invert_in_darkmode" align=middle width="43.50555pt" height="24.56552999999997pt"/> is a vector of `num_classes` dimension. Each entry in the vector tells the probability that the input belongs to that class.

For example, if our classification problem concerns cats, dogs and cows. Then `prediction[1]` tells the probability that in input is a dog.

### Intuition for the averaging
Why does it help to sample many parameter vectors and average them?

Three types of intuition:

  * _Intuition_: This averaging looks like an ensemble method. More models know more than one model.
  * _Robust_: Think about the adversarial examples. An image might be an adversarial input for one model, but it is hard to be adversarial for all the models, so we average out this adversarial prediction.
  * _Formal_: This sampling and averaging approximates the posterior predictive distribution: <img alt="$p(y|x) = \int_w p(p|x,w)p(w|data)dw$" src="https://rawgit.com/RobRomijnders/weight_uncertainty/master/svgs/dbb647613f36e527b72e9cdccbd6c3ce.svg?invert_in_darkmode" align=middle width="239.23399499999996pt" height="26.48447999999999pt"/>

(When I say _different models_, I mean to say: our model with different parameter vectors.)


# Getting the uncertainty

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

Intuitively, we drop the parameters first that are least useful. For example, if a parameter has a high posterior probability of being zero, we might as well drop it. Conversely, if a parameter has a low posterior probability of being zero, we want to keep it. We follow this intuition as we prune parameters: 1) we pick a threshold for the zero probability and 2) we sweep over all the parameters and drop the ones whose probability at zero is above the threshold. 

Again, PyData loves python, so let's write some python

```python
for param, mu, sigma in approximation():
    zero_probability = normal.pdf(mu, sigma, 0.0)
    if zero_probability > threshold:
        model.drop(param)
```
For the corresponding code in the project, see: `RestoredModel.pruning(threshold)`

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

Examples and the uncertainty curves are in the [presentation](https://github.com/RobRomijnders/weight_uncertainty/blob/master/docs/presentation/versions/final_pydata18_bayes_nn_rob_romijnders_1.pdf)

### CIFAR10
Pruning curve
![pruning_curve_cifar](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/pruning_curves/cifar_pruning_curve.png?raw=true)

Examples and the uncertainty curves are in the [presentation](https://github.com/RobRomijnders/weight_uncertainty/blob/master/docs/presentation/versions/final_pydata18_bayes_nn_rob_romijnders_1.pdf)

### ECG5000
Pruning curve
![pruning_curve_ucr](https://github.com/RobRomijnders/weight_uncertainty/blob/master/weight_uncertainty/im/pruning_curves/ucr_pruning_curve.png?raw=true)

Examples and the uncertainty curves are in the [presentation](https://github.com/RobRomijnders/weight_uncertainty/blob/master/docs/presentation/versions/final_pydata18_bayes_nn_rob_romijnders_1.pdf)


# Summary
Our motivation for this project concerns two problems with neural networks: uncertainty and pruning. Conventional neural networks use one parameter vector. We use the posterior and sample many parameter vectors. For a prediction, we average the output of the neural net with each parameter vector. We find the uncertainty as the entropy of the posterior predictive distribution. We prune parameters whose probability of being zero exceeds a threshold. Our experiment show that we can prune 90% of the parameters while maintaining performance. We also show pictures to get intuition for our uncertainty numbers.

Our experiment are small. [This paper](https://arxiv.org/abs/1705.08665) does more extensive speed comparisons. [This paper](https://arxiv.org/pdf/1711.08244.pdf) shows how the uncertainty increases under stronger adversarial attacks.

I hope that this code is useful to you. Contact me at romijndersrob@gmail.com if I can help more. 
(Please understand that I get many emails: Formulate a concise question)


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
