import pickle
import streamlit as st
import numpy as np

# Set up dynamic file paths
model_path = "C:/final year project/trained_model (1).sav"
scaler_path = "C:/final year project/scaler.sav"

# Load the saved models
try:
    diabetes_model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    st.info('an apple a day keeps the doctor away.')
except Exception as e:
    st.error(f'Error loading models: {e}')



# sidebar for navigation
with st.sidebar:
    selected = st.selectbox('Diabetes Prediction System', ['Diabetes Prediction'])

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction application')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Enter no. of Pregnancies')

    with col2:
        Glucose = st.text_input('Your Glucose Level')

    with col3:
        BloodPressure = st.text_input('Your Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value(Fat)')

    with col2:
        Insulin = st.text_input('Your Insulin Level')

    with col3:
        BMI = st.text_input('Your BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Check Result'):
        try:
            # Convert input data to appropriate data type
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

            # Reshape the array for prediction
            input_data_reshaped = np.asarray([input_data])

            # Standardize input data using the saved scaler
            input_data_scaled = scaler.transform(input_data_reshaped)

            diab_prediction = diabetes_model.predict(input_data_scaled)

            if diab_prediction[0] == 0:
                diab_diagnosis = 'Oops..!! you are diabetic. Please visit your nearest clinic.'
            else:
                diab_diagnosis = 'Congratulations..!! you are not diabetic. Enjoy the day.'

            # Display input values along with the diagnosis result
            st.text(f'Input Values: {input_data}')
            st.success(diab_diagnosis)

        except ValueError:
            st.error('Please enter valid numerical values for all input fields.')
        except Exception as ex:
            st.error(f'Error during prediction: {ex}')
