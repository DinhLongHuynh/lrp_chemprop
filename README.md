# OPTIMIZING TRAINING SETTING FOR EXPLAINABLE CHEMPROP MODEL USING LAYER-WISE RELEVANCE PROPAGATION

## About this project
Hello, this repository showcases the work for my Master's Thesis at Uppsala University under the supervision of Dr. Christian Sköld. The project's goal is to investigate the optimal training settings for an explainable ChemProp model. It utilizes Layer-wise Relevance Propagation (LRP) as an explainable AI method within the Directed Message Passing Neural Network architecture. This repository contains all the code used to implement LRP and build the evaluation framework.

<p align="center">
  <img src="images/project_cover.png" alt="Alt text" width="700">
</p>


## Features
- Implementation of Layer-wise Relevance Propagation for Chemprop - Message Passing Neural Networks.
- Visualization tools for atom/bond contribution analysis.
- Evaluation framework for faithfulness and correctness of explanations.

## Tutorials
To dicover how to use lrp_chemprop module for analysing explainability of chemprop model, please check: https://github.com/DinhLongHuynh/lrp_chemprop/blob/main/examples/LRP_analysis.ipynb


## Installation

```bash
# Clone the repository
git clone https://github.com/DinhLongHuynh/lrp_chemprop.git
cd lrp_chemprop

# Create and activate a conda environment
conda create -n lrp_chemprop python=3.8
conda activate lrp_chemprop

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```


## Acknowledgements
- [Chemprop](https://github.com/chemprop/chemprop) for the base molecular property prediction framework.
