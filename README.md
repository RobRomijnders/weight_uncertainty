# weight_uncertainty

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style={0}&logo=scikit-learn&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)

---

## Overview

This project provides a way to build more reliable machine learning models by quantifying uncertainty in predictions. It allows developers to understand when a model is unsure, potentially improving decision-making and robustness in real-world applications where confidence levels are crucial.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---
## Core features

1. **Bayesian Neural Network Implementation**: The project implements a Bayesian neural network using the 'Bayes by Backprop' method, allowing for uncertainty estimation in model predictions.
2. **Weight Uncertainty Modeling**: A core aspect of the project is modeling uncertainty directly within the neural network weights using variational inference techniques.
3. **Model Training and Evaluation**: The project provides functionality for training the Bayesian neural network and evaluating its performance on datasets like UCR, CIFAR, and MNIST.
4. **Data Loading and Preprocessing**: Includes a data loader (`Dataloader`) for handling various datasets, including options for augmentation.

---

## Installation

Install weight_uncertainty using one of the following methods:

**Build from source:**

1. Clone the weight_uncertainty repository:
```sh
git clone https://github.com/RobRomijnders/weight_uncertainty
```

2. Navigate to the project directory:
```sh
cd weight_uncertainty
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Documentation

A detailed weight_uncertainty description is available [here](https://github.com/RobRomijnders/weight_uncertainty/tree/master/docs).

---

## Contributing

- **[Report Issues](https://github.com/RobRomijnders/weight_uncertainty/issues)**: Submit bugs found or log feature requests for the project.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    RobRomijnders (2018). weight_uncertainty repository [Computer software]. https://github.com/RobRomijnders/weight_uncertainty

### BibTeX format:

    @misc{weight_uncertainty,

        author = {RobRomijnders},

        title = {weight_uncertainty repository},

        year = {2018},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/RobRomijnders/weight_uncertainty.git}},

        url = {https://github.com/RobRomijnders/weight_uncertainty.git}

    }

---
