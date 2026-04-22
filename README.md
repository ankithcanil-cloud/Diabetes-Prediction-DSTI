# Diabetes Prediction Using Machine Learning

## Project Overview
This project predicts the likelihood of diabetes in patients based on various health metrics using a **Random Forest Classifier**. The model is deployed using **Streamlit** for real-time predictions.

## Dataset
The dataset is sourced from the **TAIPEI Diabetes Database** and contains health indicators such as glucose levels, BMI, insulin levels, and more.

## Features Used
- **Pregnancies**
- **Plasma Glucose Concentration**
- **Diastolic Blood Pressure**
- **Triceps Skin Fold Thickness**
- **Serum Insulin**
- **Body Mass Index (BMI)**
- **Diabetes Pedigree Function**
- **Age**
## Features Contains
- **Pregnancies: Number of pregnancies the patient has had.**

- **Plasma Glucose Concentration: The plasma glucose concentration after 2 hours in an oral glucose tolerance test.**

- **Diastolic Blood Pressure: The diastolic blood pressure (mm Hg).**

- **Triceps Skin Fold Thickness: The triceps skin fold thickness (mm).**

- **Serum Insulin: The serum insulin level (mu U/ml).**

- **Body Mass Index (BMI): The body mass index of the patient.**

- **Diabetes Pedigree Function: A function that represents the genetic relationship of the patient to diabetes (a value based on the patient's family history).**

- **Age: Age of the patient (in years).**


## Installation

1. Clone the repository
First, clone the repository to your local machine:
   ```bash
   git clone https://github.com/Surya9810/Diabetes-Prediction-DSTI.git
   cd Diabetes-Prediction-DSTI

2. Install Dependencies
Once you've cloned the repository, install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
This will install all the libraries required to run the app, including joblib, streamlit, and others.

3. Run the Streamlit App
To launch the app, run the following command:
   ```bash
   streamlit run app.py
This will open the app in your default browser, allowing you to interact with the model and make predictions in real time.

4. Model Training (Jupyter Notebook)
The model training process is handled within a Jupyter notebook. To train the model, open and run the following Jupyter notebook:
   ```bash
   jupyter notebook diabetes_ml_pipeline.ipynb
This notebook performs the following steps:

- **Preprocesses the dataset**

- **Trains the Random Forest Classifier**

- **Saves the necessary files (random_forest_model.pkl, scaler.pkl)**


## Deployment

The app can be deployed using platforms like Streamlit Share or Heroku.

1. Push the code to GitHub (already done).

2. Connect the repository to a hosting platform.

3. Run the app on the server using:
   ```bash
   streamlit run app.py

## Results

-- **Model Accuracy: ~93.27%**

-- **ROC-AUC Score: 0.9812**

-- **Best Model: Random Forest Classifier**

## License

This project is licensed under the MIT License.

## Environment, Packages, and Libraries

The following packages are used in this project:

- **Streamlit: For deploying the app.**

- **scikit-learn: For machine learning model (Random Forest Classifier).**

- **joblib: For saving and loading the trained model.**

- **pandas: For data manipulation.**

- **numpy: For numerical operations.**

1. To install the required libraries, simply run:
   ```bash
   pip install -r requirements.txt

## References

- **TAIPEI Diabetes Dataset**
- **Scikit-learn Documentation**
