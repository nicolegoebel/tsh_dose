import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_squared_error

st.title('For your new recommended TSH Dose, please enter the following data:')

# get model data
@st.cache_data
def load_data(fname = "ForAnalysis.xlsx"):
        df = pd.read_excel(fname, header=1)
        return df.dropna(how="any")


# model
@st.cache_resource
def linear_regression_model_basic(
        df, 
        input_cols=['weight', 'Initial weekly dose', "TSH1", "TSH2"], 
        output_col='New Dose'
        ):
    y = df[output_col].values
    X = df[input_cols].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print(f"r^2: {linreg.score(X_test, y_test): .2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred): .2f}")
    print(f"Coefficients: {linreg.coef_}")
    print(f"Intercept: {linreg.intercept_}")

    return linreg, X_train, X_test, y_train, y_test

def predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH2):
    return model.predict([[float(weight), float(initial_weekly_dose), float(TSH1), float(TSH2)]])[0]

# load data
data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('Done! (using cached data)')

# run model
model, _, _, _, _ = linear_regression_model_basic(df)

# predict new dose

weight = st.number_input("Enter your weight in Kg")#, value='float', step='float', min_value=40., max_value=150.)
initial_weekly_dose = st.number_input("Enter your weekly dose (in ml?)")#, value='float',  step='float', min_value=100., max_value=1500.)
TSH1 = st.number_input("Enter your original TSH level")#, value='float', step='float', min_value=0., max_value=15.)
TSH2 = st.number_input("Enter your new TSH level")#, value='float', step='float', min_value=0., max_value=15.)

new_dose = predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH2)

output = st.write("Your new dose is:", new_dose)










