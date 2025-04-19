# Binary Image Classification (Cats vs Dogs) using CNN and Transfer Learning

## 🧠 Problem Statement

Classify images as either **cat** or **dog** using deep learning techniques.  
This project aims to:
- Build a **Convolutional Neural Network (CNN)** from scratch for binary image classification.
- Apply **transfer learning** using a pre-trained model such as **MobileNetV2** or **VGG16**.
- **Compare accuracy** between the custom CNN and the transfer learning models and predict the cat & dog images accurately.

## 🔍 Explanation

This notebook (`A2_29_P9.ipynb`) includes the full implementation of a binary image classification model using the following steps:

1. **Data Preprocessing**: 
   - Load and prepare the Cats vs Dogs dataset.
   - Normalize and augment the image data for better model generalization.

2. **Model 1: CNN from Scratch**:
   - Construct a CNN architecture using TensorFlow/Keras.
   - Train the model on the dataset and evaluate its accuracy.

3. **Model 2: Transfer Learning**:
   - Use pre-trained models such as **MobileNetV2** or **VGG16** with fine-tuning.
   - Train the model on the same dataset.
   - Evaluate and compare the results with the custom CNN.

4. **Comparison**:
   - Compare the performance of the models in terms of:
     - CNN accuracy & loss.
     - Transfer Learning accuracy & loss.

5. **Prediction and Evaluation**:

   - A unified prediction function (predict_image(model, img_path)) is used to test any trained model (CNN or transfer learning).
   - The function: 
      - Loads and preprocesses an image.
      - Makes a prediction using the given model.
      - Displays the image with predicted label (cat or dog) and prediction confidence.
      - Uses emoji for clarity and visualization (🐱 for cat, 🐶 for dog).
   - The predicted class is based on a sigmoid threshold of 0.5:
      - prediction > 0.5 → cat
      - prediction <= 0.5 → dog



## 📂 Dataset

This project uses the **Dogs vs. Cats** dataset provided by Microsoft and available on Kaggle.

- 📥 [Download from Kaggle](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset)

To use the dataset:
1. Sign in to your [Kaggle account](https://www.kaggle.com/).
2. Navigate to the competition page linked above.
3. Download and extract the dataset files.

## 📁 File Included

- `A2_29_P9.ipynb`: Main Jupyter notebook with complete code and analysis.
