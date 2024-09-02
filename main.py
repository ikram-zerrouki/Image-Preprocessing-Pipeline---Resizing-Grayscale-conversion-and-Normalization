import os
import numpy as np
import cv2

# Directories for input and output images
input_dir = os.path.join(os.path.dirname(__file__), 'Dataset', 'images')
output_dir = os.path.join(os.path.dirname(__file__), 'Processed_Dataset', 'images')

# Create directories to save processed images if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to preprocess the image (resize, grayscale conversion, normalization)
def preprocess_image(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

        # Convert to grayscale if not already
        if len(image.shape) == 3:  # If image has 3 channels (BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to 224x224
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize the image (scale pixel values to [0, 1])
        normalized_image = resized_image / 255.0

        return normalized_image
    except ValueError as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process images
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        processed_image = preprocess_image(image_path)
        if processed_image is not None:
            processed_image_path = os.path.join(output_dir, filename)
            processed_image_uint8 = (processed_image * 255).astype(np.uint8)
            cv2.imwrite(processed_image_path, processed_image_uint8)

print("Image preprocessing completed and images saved.")
