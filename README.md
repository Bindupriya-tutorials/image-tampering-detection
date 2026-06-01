# Image Forgery Detection using CASIA 2.0 Dataset

A deep learning-based image tampering detection project trained using the **CASIA 2.0 Image Tampering Detection Dataset**.

## Overview

This project detects whether an image is:

- Real (Authentic)
- Fake (Tampered)

The model was trained using deep learning techniques for image forgery detection.

## Features

- Detects tampered (fake) images
- Classifies images as Real or Fake
- Deep learning-based prediction
- Trained using CASIA 2.0 dataset
- Easy model retraining using `train.py`

## Dataset Used

**CASIA 2.0 Image Tampering Detection Dataset**

Dataset contains:

- Authentic (Real) Images
- Tampered (Fake) Images

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Flask

## Project Structure

```txt
project-folder/
│── app.py
│── train.py
│── model/
│    └── forgery_model.h5
```
## Model File

The trained model file (`forgery_model.h5`) is not uploaded because of GitHub file size limitations.

You can train the model again using:

```bash
python train.py
```

Or place the trained `.h5` model file inside:

```txt
model/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Project

```bash
python app.py
```

## Author
Bindu
