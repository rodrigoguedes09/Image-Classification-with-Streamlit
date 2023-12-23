# Image Classification with Streamlit and InceptionV3
In the dynamic landscape of machine learning and computer vision, image classification has emerged as a powerful application, enabling systems to understand and categorize visual data. This tutorial explores the fusion of two robust technologies: Streamlit, a user-friendly Python library for creating interactive web applications, and InceptionV3, a state-of-the-art convolutional neural network (CNN) renowned for its accuracy in image recognition tasks.

## Overview
This is a simple app created with Streamlit to perform image classification using the pre-trained InceptionV3 model.
![image_2023-11-23_130403673](https://github.com/rodrigoguedes09/streamlit-initial/assets/61996985/c3cad6f5-0d56-4909-acb4-b013117c4267)

## Prerequisites
Make sure you have Python and pip installed on your system.

## Installation
Clone this repository to your local environment.
```bash
git clone https://github.com/rodrigoguedes09/steamlit-initial.git
cd your-repository
```

## Install the dependencies
```bash
pip install -r requirements.txt
```

## Running
```bash
streamlit run app.py
```
![image_2023-11-23_130446486](https://github.com/rodrigoguedes09/streamlit-initial/assets/61996985/56492748-f8cf-478c-b5d7-0514b1511475)

## Usage
In the app, upload an image using the "Choose an image..." button.
The app will display the uploaded image and the top three predicted classes by the InceptionV3 model.

## Dependencies
streamlit
pillow
tensorflow
keras

#Notes
Make sure to have an internet connection when running the app for the first time, as the InceptionV3 model will be downloaded automatically.
This app is an educational example and can be extended to handle more complex cases or include additional features.

