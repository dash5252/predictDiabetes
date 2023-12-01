import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the saved model
loaded_model = pickle.load(open("C:/final year project/trained_model (1).sav", 'rb'))

# Load the scaler used during training
scaler = pickle.load(open("C:/final year project/scaler.sav", 'rb'))

# Creating a function for Prediction
def diabetes_prediction(input_data):
    # Convert input data to appropriate data type
    input_data = [float(value) for value in input_data]

    # Reshape the array for prediction
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)

    # Standardize input data using the same scaler used during training
    input_data_scaled = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_scaled)

    if prediction[0] == 0:
        return '''The person is not diabetic\n
    please follow these suggestions:\n 
    1. Schedule regular checkups, especially if you have risk factors for diabetes such as a family history, obesity, or a sedentary lifestyle.\n
    2. Adopt healthy eating habits.\n
    3. Stay physically and mentally active.'''
    else:
        return '''The person is diabetic\n
    please follow these suggestions:\n
    1. Follow the treatment/medication plan as mentioned by your doctor.\n
    2. Monitor blood sugar levels regularly or with the particular interval.\n
    3. Embrace a balanced and healthy lifestyle.'''

def main():
    # Giving a title
    st.title('Predicting a Diabetes')
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_data)
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
