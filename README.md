# Plant Disease Class Prediction

This project is a web application built with Streamlit and TensorFlow that predicts the disease class of a plant leaf from an uploaded image. The app uses a pre-trained Keras model to classify images into various plant disease categories.

## Features

- Upload an image of a plant leaf (jpg, jpeg, png).
- Predict the disease class of the plant leaf using a trained deep learning model.
- Display the uploaded image alongside the prediction result.
- User-friendly interface with instructions.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

This will open the app in your default web browser. Upload an image of a plant leaf to get the disease class prediction.

## Model

The app uses a pre-trained Keras model saved as `plant_disease_model.keras`. The model classifies images into multiple plant disease categories including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato diseases.

## Dependencies

- streamlit==1.24.1
- tensorflow==2.13.0
- numpy==1.24.4
- pillow==10.0.0

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed with Streamlit and TensorFlow.
