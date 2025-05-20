# Custom CNN for Pneumonia Detection from Chest X-Rays

A Streamlit web application to classify chest X-ray images as 'NORMAL' or 'PNEUMONIA' using a custom-trained Convolutional Neural Network (CNN).

## Features
- Upload X-ray images (JPG, JPEG, PNG).
- Custom CNN model trained from scratch.
- Interactive UI built with Streamlit.
- Displays prediction and confidence scores.

## Model
- The model `custom_cnn_pneumonia_model.keras` is a custom CNN.
- Input image size: 150x150 pixels.

## Setup

1.  **Install Git LFS (if your model file is >100MB):**
    *This project's model might be small enough, but good practice if it grows.*
    ```bash
    git lfs install
    ```

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git
    cd YourRepositoryName
    ```
    *(Replace `YourUsername/YourRepositoryName` with your actual repo details)*

3.  **Create and activate a Python virtual environment (e.g., using Python 3.11+):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application
```bash
streamlit run app.py