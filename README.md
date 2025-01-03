# MaskCycleGAN-VC
About  Implementation of Kaneko et al.'s MaskCycleGAN-VC model for non-parallel voice conversion.


# Setup
Clone the repository.

```
git clone git@github.com:DJBOYBLACKPINK/MaskCycleGAN-VC.git
cd MaskCycleGAN-VC
```

Create the conda environment.
```
conda env create -f environment.yml
conda activate MaskCycleGAN-VC
```

# NKRAFA Thai Dataset

# Data Preprocessing

# Training

# Testing (Inference)

# Code Organization
```
├── README.md                       <- Top-level README.
├── environment.yml                 <- Conda environment
├── .gitignore
├── LICENSE
|
├── args
│   ├── base_arg_parser             <- arg parser
│   ├── train_arg_parser            <- arg parser for training (inherits base_arg_parser)
│   ├── cycleGAN_train_arg_parser   <- arg parser for training MaskCycleGAN-VC (inherits train_arg_parser)
│   ├── cycleGAN_test_arg_parser    <- arg parser for testing MaskCycleGAN-VC (inherits base_arg_parser)
│
├── bash_scripts
│   ├── mask_cyclegan_train.sh      <- sample script to train MaskCycleGAN-VC
│   ├── mask_cyclegan_test.sh       <- sample script to test MaskCycleGAN-VC
│
├── data_preprocessing
│   ├── preprocess_vcc2018.py       <- preprocess VCC2018 dataset
│
├── dataset
│   ├── vc_dataset.py               <- torch dataset class for MaskCycleGAN-VC
│
├── logger
│   ├── base_logger.sh              <- logging to Tensorboard
│   ├── train_logger.sh             <- logging to Tensorboard during training (inherits base_logger)
│
├── saver
│   ├── model_saver.py              <- saves and loads models
│
├── mask_cyclegan_vc
│   ├── model.py                    <- defines MaskCycleGAN-VC model architecture
│   ├── train.py                    <- training script for MaskCycleGAN-VC
│   ├── test.py                     <- training script for MaskCycleGAN-VC
│   ├── utils.py                    <- utility functions to train and test MaskCycleGAN-VC

```

# Acknowledgements
This repository was inspired by [GANtastic3](https://github.com/GANtastic3)'s implementation of [MaskCycleGAN-VC](https://github.com/GANtastic3/MaskCycleGAN-VC).
