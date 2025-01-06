# CLAHE Implementation
This repository contains a custom implementation of CLAHE (Contrast Limited Adaptive Histogram Equalization) in Python using OpenCV and NumPy. CLAHE is a technique used to enhance the contrast of an image, particularly in regions of varying brightness.

# #  Features
• Custom CLAHE Algorithm
Implements CLAHE from scratch, including interpolation and histogram clipping.
Supports both grayscale and color images.
• Comparison with OpenCV's CLAHE
Provides side-by-side results of the custom CLAHE implementation and OpenCV's built-in CLAHE function.

# #  Requirements
• Python 3.x
• OpenCV
• NumPy
• Matplotlib
code: pip install numpy opencv-python matplotlib

## Usage
•  Clone this repository:
git clone https://github.com/yourusername/clahe-implementation.git
cd clahe-implementation
python clahe.py
Place your input image (1.jpg) in the same directory
The script reads an image, applies CLAHE using both custom and OpenCV implementations, and displays the results.

