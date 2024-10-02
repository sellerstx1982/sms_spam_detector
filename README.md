# SMS Spam Classification with Gradio and Scikit-Learn

This repository contains a Jupyter Notebook for building a machine learning model to classify SMS messages as "spam" or "not spam". The notebook uses the Scikit-Learn library to train a text classification model and the Gradio library to provide an interactive interface for testing the model.

## Overview

The notebook demonstrates the process of building an SMS spam classification model using:
- **TF-IDF Vectorization** to convert text data into numerical features.
- **Linear Support Vector Classifier (LinearSVC)** for training the model.
- **Gradio** for creating an interactive user interface to test the model in real-time.

## Dataset

The dataset used for training the model is the **SMSSpamCollection** dataset, which contains labeled text messages ("spam" or "ham"). It is loaded into a pandas DataFrame from the file `Resources/SMSSpamCollection.csv`.

## Features

- **Text Preprocessing**: Text messages are transformed using TF-IDF vectorization to normalize and convert them into feature vectors.
- **Model Training**: The model is trained using a Linear Support Vector Classifier.
- **Interactive Prediction**: An interactive Gradio interface is built for users to input their own SMS messages and receive real-time predictions.

## Code Summary

1. **Dependencies Setup**: The notebook starts by importing necessary libraries such as `pandas`, `scikit-learn`, and `gradio`.
2. **Data Loading**: The SMS dataset is loaded into a DataFrame.
3. **Model Training**: The `sms_classification` function is used to create a TF-IDF and LinearSVC pipeline, which is then trained on the dataset.
4. **SMS Prediction**: The `sms_prediction` function takes a user-provided SMS message and returns whether it is classified as "spam" or "not spam".

## Usage

To use the notebook:

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    ```
2. **Install Dependencies**:
    Ensure you have `pandas`, `scikit-learn`, and `gradio` installed. You can install them via pip:
    ```bash
    pip install pandas scikit-learn gradio
    ```
3. **Run the Notebook**:
    Open the notebook in Jupyter and run the cells step-by-step to train the model and interact with the Gradio interface.

## Gradio Interface

The notebook uses Gradio to create an easy-to-use web interface. Once you run the notebook, a web app will be generated where you can type in an SMS message and get a prediction on whether it is spam or not.

## Example

After training the model, you can use the `sms_prediction` function via the Gradio interface to classify any SMS message:

- Input: `"Congratulations! You've won a $1,000 gift card! Click here to claim."`
- Output: `The text message is classified as spam.`

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- gradio

## Conclusion

This notebook is a practical example of how to build a text classification model using Scikit-Learn and how to make it accessible via a user-friendly web interface with Gradio. It demonstrates end-to-end model training, evaluation, and deployment in a simple and interactive way.

## Acknowledgements

- Dataset: [UCI Machine Learning Repository - SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Libraries: Scikit-Learn, Gradio

## License

This project is licensed under the MIT License.
