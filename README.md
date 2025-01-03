# MaskCycleGAN-VC
About  Implementation of Kaneko et al.'s MaskCycleGAN-VC model for non-parallel voice conversion.


# Setup

Operating System.

```
Ubuntu 22.04.4 LTS 64-bit
```

Install Anaconda to create a Python environment for the model.

```
https://accuweb.cloud/resource/articles/install-anaconda-in-ubuntu-22-04-tutorial-for-beginners
```

Install CUDA Toolkit for using NVIDIA GPU’s CUDA capabilities.

```
https://www.cherryservers.com/blog/install-cuda-ubuntu
```

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

The authors of the paper used the dataset from the Spoke task of Navaminda Kasatriyadhiraj Royal Air Force Academy (NKRAFA). This is a dataset of non-parallel utterances from 3 male and 2 female speakers. Each speaker utters approximately 135 sentences.

Download the dataset from the command line.
```
wget --no-check-certificate https://googledrive.com/NKRAFA_Thai_training.zip?sequence=2&isAllowed=y
wget --no-check-certificate https://googledrive.com/NKRAFA_Thai_evaluation.zip?sequence=3&isAllowed=y
```

Unzip the dataset file.
```
mkdir NKRAFA_Thai
apt-get install unzip
unzip NKRAFA_Thai_training.zip?sequence=2 -d NKRAFA_Thai/
unzip NKRAFA_Thai_evaluation.zip?sequence=3 -d NKRAFA_Thai/
```

# Data Preprocessing

To expedite training, we preprocess the dataset by converting waveforms to melspectograms, then save the spectrograms as pickle files `<speaker_id>normalized.pickle` and normalization statistics (mean, std) as npz files `<speaker_id>_norm_stats.npz`. We convert waveforms to spectrograms using a [melgan vocoder](https://github.com/descriptinc/melgan-neurips) to ensure that you can decode voice converted spectrograms to waveform and listen to your samples during inference.

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory NKRAFA_Thai/training \
  --preprocessed_data_directory NKRAFA_Thai_preprocessed/training \
  --speaker_ids SM01 TM02 TM03 TF04 TF05
```

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory NKRAFA_Thai/evaluation \
  --preprocessed_data_directory NKRAFA_Thai_preprocessed/evaluation \
  --speaker_ids SM01 TM02 TM03 TF04 TF05
```

# Training

Train MaskCycleGAN-VC to convert between `<speaker_A_id>` and `<speaker_B_id>`. You should start to get excellent results after only several hundred epochs.
```
python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_<speaker_id_A>_<speaker_id_B> \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir NKRAFA_Thai_preprocessed/training/ \
    --speaker_A_id <speaker_A_id> \
    --speaker_B_id <speaker_B_id> \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 1600 \
    --batch_size 1 \
    --discriminator_lr 5e-4 \
    --generator_lr 2e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
```

Example 

```
python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_SM01_TM02 \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir NKRAFA_Thai_preprocessed/training/ \
    --speaker_A_id SM01 \
    --speaker_B_id TM02 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 1600 \
    --batch_size 1 \
    --discriminator_lr 5e-4 \
    --generator_lr 2e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
```

To continue training from a previous checkpoint in the case that training is suspended, add the argument `--continue_train` while keeping all others the same. The model saver class will automatically load the most recently saved checkpoint and resume training.

Launch Tensorboard in a separate terminal window.
```
tensorboard --logdir results/logs
```

# Testing / Inference / Evaluation

Test your trained MaskCycleGAN-VC by converting between `<speaker_A_id>` and `<speaker_B_id>` on the evaluation dataset. Your converted .wav files are stored in `results/<name>/converted_audio`.

```
python -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_<speaker_A_id>_<speaker_B_id> \
    --save_dir results/ \
    --preprocessed_data_dir NKRAFA_Thai_preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id <speaker_A_id> \
    --speaker_B_id <speaker_B_id> \
    --ckpt_dir ~/Desktop/MaskCycleGAN-VC/results/mask_cyclegan_vc_<speaker_A_id>_<speaker_B_id>/ckpts \
    --load_epoch 1600 \
    --model_name generator_A2B \
```

Example 

```
python -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_SM01_TM02 \
    --save_dir results/ \
    --preprocessed_data_dir NKRAFA_Thai_preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id SM01 \
    --speaker_B_id TM02 \
    --ckpt_dir ~/Desktop/MaskCycleGAN-VC/results/mask_cyclegan_vc_SM01_TM02/ckpts \
    --load_epoch 1600 \
    --model_name generator_A2B \
```

Toggle between A->B and B->A conversion by setting `--model_name` as either `generator_A2B` or `generator_B2A`.

Select the epoch to load your model from by setting `--load_epoch`.


# Code Organization
```
~Desktop/MaskCycleGAN-VC
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
│   ├── preprocess_vcc2018.py       <- preprocess NKRAFA Thai Dataset
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
