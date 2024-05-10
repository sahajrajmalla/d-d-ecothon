import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

temp_model = pickle.load(open('temp_model.pkl', 'rb'))
pressure_model = pickle.load(open('pressure_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def calculate_molality(TDS, molar_mass_NaCl=58.44):
    return TDS / molar_mass_NaCl

def calculate_energy(initial_temp_K, final_temp_K, mass=1, c=4186):
    delta_T = final_temp_K - initial_temp_K
    return mass * c * delta_T

def predict_temperature(initial_temp_K, TDS):
    molality = calculate_molality(TDS)
    final_temp_K_simulated = initial_temp_K + 10
    energy = calculate_energy(initial_temp_K, final_temp_K_simulated)
    input_data = pd.DataFrame({
        'TDS': [TDS],
        'Initial_Temperature_K': [initial_temp_K],
        'Molality': [molality],
        'Energy_Consumed_J': [energy]
    })
    scaled_input = scaler.transform(input_data)
    return temp_model.predict(scaled_input)[0]

def predict_pressure(initial_temp_K, TDS):
    molality = calculate_molality(TDS)
    final_temp_K_simulated = initial_temp_K + 10
    energy = calculate_energy(initial_temp_K, final_temp_K_simulated)
    input_data = pd.DataFrame({
        'TDS': [TDS],
        'Initial_Temperature_K': [initial_temp_K],
        'Molality': [molality],
        'Energy_Consumed_J': [energy]
    })
    scaled_input = scaler.transform(input_data)
    return pressure_model.predict(scaled_input)[0]

def main():
    st.title('Temperature and Pressure Prediction Dashboard')

    st.sidebar.header('Input Parameters')
    initial_temp_C = st.sidebar.number_input('Initial Temperature (Celsius)', min_value=-273, value=25)
    TDS = st.sidebar.number_input('Total Dissolved Solids (TDS) in g/kg', min_value=0, value=100)

    initial_temp_K = initial_temp_C + 273.15

    if st.sidebar.button('Predict Outcomes'):
        predicted_temp_K = predict_temperature(initial_temp_K, TDS)
        predicted_temp_C = predicted_temp_K - 273.15
        predicted_pressure = predict_pressure(initial_temp_K, TDS)

        bias = np.random.uniform(-0.05, 0.05)
        predicted_pressure -= (predicted_pressure * bias) - (1000 * bias) 

        predicted_pressure = max(0, min(predicted_pressure, 101000))

        st.subheader('Prediction Results:')
        st.write(f'**Final Temperature (Celsius):** {predicted_temp_C:.2f}')
        st.write(f'**Pressure (Pa):** {predicted_pressure:.2f}')
        st.write(f'**Energy Conserved (Joules):** {calculate_energy(initial_temp_K, predicted_temp_K):.2f}')

    if st.checkbox('Show Raw Data'):
        st.subheader('Raw Data')
        data = pd.read_csv('data.csv')
        st.write(data)

if __name__ == "__main__":
    main()
