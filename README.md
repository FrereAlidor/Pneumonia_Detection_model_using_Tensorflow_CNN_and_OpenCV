# Pneumonia_Detection_model_using_Tensorflow_CNN_and_OpenCV


This project implements a deep learning model to detect pneumonia in chest X-ray images. The model is built using **TensorFlow** and **Convolutional Neural Networks (CNN)**, along with **OpenCV** for image preprocessing. The dataset used for this project is sourced from Kaggle's **Chest X-ray Pneumonia dataset**.

## Dataset

The dataset used in this project can be found on Kaggle: [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It consists of X-ray images divided into two categories:
- **Normal** (Healthy lung images)
- **Pneumonia** (Infected lung images)

## Folder Structure

```
Pneumonia_Detection_model_using_Tensorflow_CNN_and_OpenCV/
│
├── utils/
│   ├── config/            # Configuration files for the project
│   ├── conv_bc_model/     # Custom model architecture and checkpoints
│   ├── create_dataset/    # Scripts to load and prepare the dataset
│   └── getPaths/          # Scripts to retrieve image paths
│
├── Detect_Pneumonia.py    # Script to run the trained model and detect pneumonia
├── train_CustomModel_16_conv_modelCheckpoint_reshuffle_data.py   # Training with custom model
├── train_MobileNet_16_modelCheckpoint_reshuffle_data.py          # Training with MobileNet model
│
├── sampleTest_Pictures/   # Sample test images for model evaluation
│   ├── Normal/            # Normal chest X-rays
│   └── Pneumonia/         # Pneumonia chest X-rays
│
├── Output/                # Output folder for model predictions and logs
└── README.md              # Project documentation
```

## How to Run

### Prerequisites
- Google Colab (recommended) or any local machine with TensorFlow and OpenCV installed
- Python 3.x
- Required libraries:
  ```bash
  pip install tensorflow opencv-python numpy matplotlib
  ```

### Training the Model

1. **Custom CNN Model**:
   To train a custom CNN model, run the `train_CustomModel_16_conv_modelCheckpoint_reshuffle_data.py` script:
   ```bash
   python train_CustomModel_16_conv_modelCheckpoint_reshuffle_data.py
   ```

2. **MobileNet Model**:
   To train using the MobileNet architecture, run the `train_MobileNet_16_modelCheckpoint_reshuffle_data.py` script:
   ```bash
   python train_MobileNet_16_modelCheckpoint_reshuffle_data.py
   ```

### Detecting Pneumonia
Once the model is trained, you can run `Detect_Pneumonia.py` to use the model for pneumonia detection on new chest X-ray images:
```bash
python Detect_Pneumonia.py
```

### Sample Test Images
Sample images for testing can be found in the `sampleTest_Pictures/` folder. These images are categorized as **Normal** and **Pneumonia**.

### Outputs
Model predictions and logs are saved in the `Output/` folder.

## Acknowledgments
This project utilizes the Chest X-ray Pneumonia dataset provided by Kaggle. Special thanks to the contributors of this dataset.

Contact
mbayandjambealidor@gmail.com
