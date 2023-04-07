import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_squared_error

st.title('For your new, recommended dose, please enter the following data:')

TSH2_const_val=2.0

# get model data
@st.cache_data
def load_data(fname = "ForAnalysis.xlsx", TSH2_const_val=2.0):
        df = pd.read_excel(fname, header=1)
        df["TSH2_const"]=TSH2_const_val

        return df.dropna(how="any")


# model
@st.cache_resource
def linear_regression_model_basic(
        df, 
        input_cols=['weight', 'Initial weekly dose', "TSH1", "TSH2_const"], 
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

def predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH2_const_val=2.0):
    return model.predict([[float(weight), float(initial_weekly_dose), float(TSH1), float(TSH2_const_val)]])[0]

for key in st.session_state.keys():
    print(key)
    del st.session_state[key]


# load data
data_load_state = st.text('Loading data...')
df = load_data()
# data_load_state.text('Done! (using cached data)')

# run model
model, _, _, _, _ = linear_regression_model_basic(df)

# predict new dose
min_wt = float(df["weight"].min())
max_wt = float(df["weight"].max())
min_init = float(df["Initial weekly dose"].min())
max_init = float(df["Initial weekly dose"].max() )
min_tsh1 = float(df["TSH1"].min())
max_tsh1 = float(df["TSH1"].max() )

#weight = st.number_input("Enter your weight in Kg (48.0-140.0):", step=0.1, min_value=min_wt, max_value=max_wt)
#initial_weekly_dose = st.number_input("Enter your weekly dose (175.0-1300.0 mL):", step=0.1, min_value=min_init, max_value=max_init)
#TSH1 = st.number_input("Enter your original TSH level (1.0-13.0 mL):", step=0.1, min_value=min_tsh1, max_value=max_tsh1)weight = st.number_input("Enter your weight in Kg (48.0-140.0):", step=0.1, min_value=min_wt, max_value=max_wt)
weight = st.slider("Enter your weight in Kg:", step=0.1, min_value=min_wt, max_value=max_wt)
initial_weekly_dose = st.slider("Enter your weekly dose:", step=0.1, min_value=min_init, max_value=max_init)
TSH1 = st.slider("Enter your original TSH level:", step=0.1, min_value=min_tsh1, max_value=max_tsh1)

#new_dose = False
st.session_state["new_dose"] = np.round(predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH2_const_val), 1)

#if new_dose:
delta=round(st.session_state["new_dose"]-initial_weekly_dose, 1)
st.metric("Your new dose is:", value=f'{st.session_state["new_dose"]} mL', delta=f'{delta} mL')
