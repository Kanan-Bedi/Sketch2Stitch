# Sketch2Stitch


---

# Sketch2Stitch: Turning Sketches into Realistic Clothing Images

Sketch2Stitch is a generative AI project designed to convert clothing sketches into realistic images. This tool is intended to streamline the design process in fashion, allowing designers to visualize their ideas and experiment with customizations before finalizing a physical garment.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset Creation](#dataset-creation)
- [Model Architecture](#model-architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Optimization and Future Improvements](#optimization-and-future-improvements)
- [License](#license)

---

## Project Overview
Sketch2Stitch uses a GAN-based (Generative Adversarial Network) architecture to turn clothing sketches into photo-realistic images. This project addresses challenges in apparel design by enabling rapid visualization, prototyping, and customization of clothing designs, enhancing the creative process.

## Features
- **Sketch to Image Conversion**: Generates realistic images from basic sketches.
- **Segmentation & Inpainting (Optional)**: Adds patterns or textures to specific regions of the generated clothing.
- **Data Augmentation**: Extensive use of data augmentation improves model robustness to various sketch styles.

## Dataset Creation
The dataset is created by taking real clothing images and generating synthetic sketches using OpenCV. The process involves:
1. Converting each clothing image to grayscale.
2. Applying inversion and Gaussian blur to create a "sketch" effect.
3. Saving these synthetic sketches as input data for the GAN model.

To enhance model generalization, various augmentations (e.g., rotation, scaling, and noise addition) are applied to the synthetic sketches.

## Model Architecture
The project utilizes a **Conditional GAN (cGAN)** architecture, which includes:
- **Generator**: Converts sketches into realistic clothing images by learning the distribution of real images.
- **Discriminator**: Differentiates between real and generated images to refine the generator’s output.
  
Additionally, loss functions like **perceptual loss** and **adversarial loss** are used to enhance the quality and realism of the generated images.

## Installation and Setup

### Prerequisites
- Python 3.7+
- Libraries: `torch`, `torchvision`, `opencv-python`, `matplotlib`, `numpy`
- Optional: Hugging Face’s `diffusers` and `transformers` for segmentation and inpainting

### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/your_username/Sketch2Stitch.git
cd Sketch2Stitch
pip install -r requirements.txt
```

### Setting Up the Data
Prepare the dataset by placing your clothing images in a folder. Modify the paths in `data_prep.ipynb` to point to your images and output folders, then run the notebook to generate sketches.

## Usage

1. **Data Preparation**: Run `data_prep.ipynb` to convert clothing images into sketches.
   
2. **Training the Model**: Execute `main.py` to train the GAN model on the prepared dataset. Use the following command:
   ```bash
   python main.py
   ```

3. **Generating Images**: After training, use the trained model to generate realistic clothing images from new sketches.

4. **Optional - Pattern Customization**: Use segmentation and inpainting (refer to the `Clothes Replacement Project` notebook) to add patterns or textures to specific regions of the generated clothing images.

## Optimization and Future Improvements
1. **Data Augmentation**: To improve generalization, further augment the sketches with varied techniques to represent more realistic hand-drawn styles.
2. **Segmentation & Inpainting**: Use segmentation to isolate parts of the clothing, and apply inpainting to add patterns, enhancing customization.
3. **Advanced Architectures**: Experiment with other architectures, like StyleGAN, for potentially better quality and control over the output.

## License
This project is licensed under the MIT License. Pre-trained models may have additional restrictions based on their source.
