# Invoice Fraud Detection using Deep Learning

An end-to-end computer vision system to detect visually fraudulent invoices using transfer learning, synthetic data generation, and fine-tuning. The project covers the full pipeline — from data preparation to model training and deployment as a web application.

---

## Overview

Invoice fraud often involves **visual tampering** such as redactions, stamps, or altered content. However, labeled fraud data is scarce.  
This project addresses that challenge by:

- Generating **synthetic fraudulent invoices**
- Training and comparing multiple **CNN-based models**
- Fine-tuning the best-performing model
- Deploying the final system as a **real-time web application**

---

## Key Features

- End-to-end ML pipeline (data → model → deployment)
- Synthetic fraud generation to overcome data scarcity
- Transfer learning with multiple CNN architectures
- Fine-tuning for improved performance
- Real-time inference via Streamlit web app

---

## Dataset

- **Base dataset:** High-quality invoice images (Kaggle)
- **Classes:**
  - `not_fraud` – original invoice images
  - `fraud` – synthetically altered invoices

### Synthetic Fraud Techniques
To create strong visual signals for the model, the following transformations were applied:
- Black bar redactions (simulating masked or altered fields)
- Large diagonal stamps such as **VOID**, **FAKE**, or **COPY**

These transformations help the model learn meaningful visual cues instead of noise.

---

## Models Used

The following pre-trained architectures were evaluated using transfer learning:

- **MobileNetV2**
- **EfficientNetB0**
- **DenseNet121**

### Training Strategy
1. Freeze base model weights
2. Train a custom classification head
3. Compare validation performance
4. Fine-tune top layers of the best-performing model using a low learning rate

**Final selected model:** MobileNetV2 (after fine-tuning)

---

## Training Details

- Image size: `224 × 224`
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Data augmentation:
  - Rotation
  - Shifts
  - Zoom
  - Shear

Early stopping and model checkpointing were used to prevent overfitting.

---

## Web Application

The trained model is deployed using **Streamlit** and allows users to:

- Upload an invoice image
- Select a trained model
- Receive a fraud / non-fraud prediction
- View confidence scores and model outputs

The app was exposed publicly using **ngrok** for testing and demonstration.

---

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Streamlit
- Kaggle
- ngrok

---

## Project Structure

├── data/
│ ├── not_fraud/
│ └── fraud/
├── models/
│ ├── MobileNetV2_best.h5
│ └── DenseNet121_best.h5
├── app.py
├── training_notebook.ipynb
└── README.md


---

## Results & Learnings

- Synthetic data generation was critical due to lack of labeled fraud data
- Subtle fraud patterns were harder to learn; stronger visual signals improved convergence
- Fine-tuning significantly improved validation accuracy
- Lightweight architectures like MobileNetV2 performed well for this task

---

## Future Improvements

- Use real-world annotated fraud data
- Extend beyond visual fraud (OCR + NLP-based checks)
- Improve robustness to subtle manipulations
- Deploy with a persistent backend instead of tunneling

---

## Notes

This project was built as a learning-focused, end-to-end system emphasizing **practical constraints, model behavior, and deployment considerations**, rather than just model accuracy.

---

## Author

Pranav Raj  
Computer Science | Machine Learning | Software Engineering



