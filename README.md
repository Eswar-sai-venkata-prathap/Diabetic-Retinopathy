# Diabetic Retinopathy Detection Pipeline

This project implements an end-to-end medical AI pipeline for detecting Diabetic Retinopathy using the IDRiD dataset.

## Features
- **Preprocessing**: CLAHE enhancement and resizing (224x224).
- **Model**: ResNet50 with specialized head (Dense 256, Dropout 0.5) trained with Weighted Categorical Cross-Entropy.
- **Explainability**: Grad-CAM heatmaps to visualize lesions.
- **Dashboard**: Streamlit app for interactive screening.

## Results
- **Confusion Matrix**: Saved in `output/plots/confusion_matrix.png`.
- **Classification Report**: Saved in `output/plots/classification_report.txt`.
- **Explainability**: Generated Grad-CAM samples in `output/plots/`.

## Execution
To launch the dashboard for interactive predictions:
```bash
streamlit run app/web_app.py
```
