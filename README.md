# Image Preprocessing Pipeline - Resizing, Grayscale Conversion, and Normalization

## Overview

This Python project provides a preprocessing pipeline for image datasets. The code performs essential preprocessing steps, 
including resizing, grayscale conversion, and normalization, to prepare images for machine learning models. Ensuring consistent image quality through preprocessing is crucial for enhancing the accuracy of AI models.

## Features

- **Resize**: Scales images to a fixed size of 224x224 pixels.
- **Grayscale Conversion**: Converts color images to grayscale if necessary.
- **Normalization**: Scales pixel values to the range [0, 1] for consistent model input.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Notes
- Ensure that the paths specified in the script match the relative paths to your dataset and output directories.
- Modify the "input_dir" and "output_dir" variables in the script as needed to fit your directory structure.
