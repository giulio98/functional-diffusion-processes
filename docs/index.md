# Continuous-Time Functional Diffusion Processes

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2303.00800-b31b1b.svg)](https://arxiv.org/abs/2303.00800)
[![NeurIPS 2023](https://img.shields.io/badge/NeurIPS-2023-4b44ce.svg)](https://nips.cc/Conferences/2023)

Authors: Giulio Franzese, Giulio Corallo, Simone Rossi, Markus Heinonen, Maurizio Filippone, Pietro Michiardi

Accepted as a poster at the 37th Conference on Neural Information Processing Systems (NeurIPS 2023).

---
## Abstract

We introduce Functional Diffusion Processes (FDPs), which generalize score-based diffusion models to infinite-dimensional function spaces. FDPs require
a new mathematical framework to describe the forward and backward dynamics, and several extensions to derive practical training objectives. These include
infinite-dimensional versions of Girsanov theorem, in order to be able to compute an ELBO, and of the sampling theorem, in order to guarantee that functional
evaluations in a countable set of points are equivalent to infinite-dimensional functions. We use FDPs to build a new breed of generative models in function spaces,
which do not require specialized network architectures, and that can work with
any kind of continuous data. Our results on real data show that FDPs achieve
high-quality image generation, using a simple MLP architecture with orders of
magnitude fewer parameters than existing diffusion models.

![Super Resolution on MNIST](assets/super_res_mnist.png)
![Samples from CELEBA](assets/samples_uvit_celeba.png)

---


## Installation

```bash
pip install git+ssh://git@github.com/giulio98/functional-diffusion-processes.git
```


## Quickstart

### Setup the Development Environment

```bash
git clone git@github.com:giulio98/functional-diffusion-processes.git
cd functional-diffusion-processes
conda env create -f env.yaml
conda activate fdp
pre-commit install
```
### Update the dependencies

```bash
pip install -e .[dev]
```

## Working with Hydra

[Hydra](https://hydra.cc/) is a framework that simplifies the configuration of complex applications, including the management of hierarchical configurations. It's particularly useful for running experiments with different hyperparameters, which is a key part of the experimentation done in this project.

With Hydra, you can easily sweep over parameters sequentially, which is demonstrated in the experiments sections below. Additionally, Hydra supports parallel experiments execution through plugins like [Joblib](https://joblib.readthedocs.io/en/latest/). This allows for concurrent execution of multiple experiment configurations, significantly speeding up the experimentation process when working with multiple hyperparameters.

---

## Setup the Project
Before you begin with any experiments, ensure to create a `.env` file with the following content:
```plaintext
export WANDB_API_KEY=<your wandb api key>
export HOME=<your_home_directory>  # e.g., /home/username
export CUDA_HOME=/usr/local/cuda
export PROJECT_ROOT=<your_project_directory>  # /home/username/functional_diffusion_processes
export DATA_ROOT=${PROJECT_ROOT}/data
export LOGS_ROOT=${PROJECT_ROOT}/logs
export TFDS_DATA_DIR=${DATA_ROOT}/tensorflow_datasets
export PYTHONPATH=${PROJECT_ROOT}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES=<your cuda devices>
```
All experiments utilize wandb for logging. However, you can opt out of using wandb by setting `trainer_logging.use_wandb=False` in the yaml files in `conf/trainers/trainer_maml` and `conf/trainers/trainer_vit`.
## Pretrained Checkpoints
In order to run the sampling and conditional generation experiments, you need to download the pretrained checkpoints.

All checkpoints are provided in this [Google Drive](https://drive.google.com/drive/folders/10-W5q5XSWXzoMktEdfX_Z9YiOG59zOZs?usp=drive_link)

Alternatively you can download them directly by running:


```bash
pip install gdown
```
```bash
gdown --id 1R9aRsV7q4yU0ey47tR7hFvKttEilUv0i
unzip logs.zip
rm logs.zip
```
# Experiments
## MNIST Experiments
Find the configurations for our paper's experiments under `conf/experiments_maml`, with corresponding scripts in `scripts/maml`.
### Training
Run the default training script, or use Hydra to experiment with hyperparameters:

```bash
# Default training
sh scripts/maml/train_mnist.sh

# Hyperparameter experimentation
python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=exp_mnist \
trainers.training_config.learning_rate=1e-5,2e-5
```
### Generation
```bash
sh scripts/maml/sample_mnist.sh
```
### Super Resolution
Run the script as-is for 128x128 resolution, or specify a different target shape:
```bash
# Default resolution
sh scripts/maml/super_resolution_mnist.sh

# Custom resolution
python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=exp_mnist_super_resolution \
samplers.sampler_config.target_shape=[512,512]
```
### FID Evaluation
```bash
sh scripts/maml/eval_mnist.sh
```
## CELEBA Experiments

### Download CELEBA Dataset
```bash
pip install gdown
```
```bash
cd ~/functional-diffusion-processes/data/tensorflow_datasets/
gdown --folder https://drive.google.com/drive/folders/1eHdU3N4Tiv6BAezAAI7LAvJTItIF8GD2?usp=share_link
```
Scripts for training and evaluating models on the CELEBA dataset are provided, using official configurations.
### Training
Train the INR or the UViT on the CELEBA dataset:
```bash
sh scripts/maml/train_celeba.sh  # for INR
sh scripts/vit/train_celeba.sh   # for UViT
```
### Generation and Conditional Generation
```bash
sh scripts/maml/sample_celeba.sh  # INR
sh scripts/vit/sample_celeba.sh   # UViT
sh scripts/maml/colorize_celeba.sh  # Colorization
sh scripts/maml/deblur_celeba.sh    # Deblurring
sh scripts/maml/inpaint_celeba.sh   # Inpainting
```
### FID Evaluation
```bash
sh scripts/maml/eval_celeba.sh  # INR
sh scripts/vit/eval_celeba.sh   # UViT
```
# Acknowledgements
Our code builds upon several outstanding open source projects and papers:
* [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde)
* [All are Worth Words: A ViT Backbone for Diffusion Models](https://github.com/baofff/U-ViT)
* [From data to functa: Your data point is a function and you can treat it like one](https://github.com/google-deepmind/functa)
* [Learning Transferable Visual Models From Natural Language Supervision](https://github.com/openai/CLIP)
* [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://github.com/GaParmar/clean-fid)

# Citation
If you use our code or paper, please cite:
```bib
@misc{franzese2023continuoustime,
      title={Continuous-Time Functional Diffusion Processes},
      author={Giulio Franzese and Giulio Corallo and Simone Rossi and Markus Heinonen and Maurizio Filippone and Pietro Michiardi},
      year={2023},
      eprint={2303.00800},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
