import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


st.title('For your new, recommended dose, please enter the following data:')

TSH2_const_val=2.0
widget="text_box"
#widget="slider"
# get model data
@st.cache_data
def load_data_add_features(fname = "ForNicole3_cleaned.csv"):   #, TSH2_const_val=2.0):
        df = pd.read_csv(fname)
        #df["TSH2_const"]=TSH2_const_val
        df['TSH_change'] = abs(df['TSH1'] - df['TSH2'])
        df['TSH_initial_high'] = (df['TSH1'] > 4.2).astype(int)
        df['TSH_initial_normal'] = ((df['TSH1'] >= 0.27) & (df['TSH1'] <= 4.2)).astype(int)
        #df['TSH_target'] = 2. - df['TSH2']
        #df['TSH_target'] = df['TSH_target'].apply(lambda x: x if x > -1.73 else x + 2)
        # Handle Missing Values in 'gender'
        missing_gender = df['gender'].isnull().sum()
        if missing_gender > 0.2 * len(df):
            df = df.drop(columns='gender')
        else:
            df['gender'].fillna(df['gender'].mode()[0], inplace=True)

        # Encode the 'gender' column if it exists
        if 'gender' in df.columns:
            label_encoder = LabelEncoder()
            df['gender'] = label_encoder.fit_transform(df['gender'].astype(str))

        return df.dropna(how="any")


# model
@st.cache_resource
def linear_regression_model_basic(
        df,
        input_cols=['weight', 'Initial Dose', "TSH1", "TSH_initial_high", "TSH_initial_normal"], #, "TSH_target"], #"TSH2_const"],
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

    linreg_all = LinearRegression()
    linreg_all.fit(X, y)
    # Perform cross-validation
    cv_scores = cross_val_score(linreg_all, X, y, cv=5, scoring='neg_mean_squared_error')

    # Calculate and print the RMSE scores
    cv_rmse_scores = np.sqrt(-cv_scores)
    cv_rmse_mean = cv_rmse_scores.mean()
    cv_rmse_std = cv_rmse_scores.std()
    print("Cross-Validation RMSE Mean:", cv_rmse_mean)
    print("Cross-Validation RMSE Std:", cv_rmse_std)

    return linreg_all, X, y, X_train, X_test, y_train, y_test

def random_forest_model(df,
                        input_cols=['weight', 'Initial Dose', "TSH1", "TSH_initial_high", "TSH_initial_normal"],
                        output_col="New Dose"):
    # Prepare the feature set and target variable
    X = df[input_cols]  #.drop(columns=['New Dose', 'TSH2']) #, 'gender'])  # Excluding the original 'gender' column
    y = df[output_col]

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X, y)

    return rf_regressor

def predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH_initial_high, TSH_initial_normal):  #, TSH2_const_val=2.0):
    new_data = {
        'weight': [weight],
        'Initial Dose': [initial_weekly_dose],
        'TSH1': [TSH1],
        'TSH_initial_high': [0],
        'TSH_initial_normal': [1],
        #'TSH_target':[2]
     }
    # Convert to DataFrame
    new_data_df = pd.DataFrame(new_data)

    # If you have a gender feature, encode it as was done during training
    # For example, if 'gender' was 'f' or 'm', and 'f' was encoded as 0
    if 'gender' in new_data_df.columns:
        new_data_df['gender'] = label_encoder.transform(['f'])


    # Make a prediction
    predicted_new_dose = model.predict(new_data_df)

    #return model.predict([[float(weight), float(initial_weekly_dose), float(TSH1), float(TSH2_const_val)]])[0]
    return predicted_new_dose[0]

for key in st.session_state.keys():
    print(key)
    del st.session_state[key]


# load data
data_load_state = st.text('Loading data...')
df = load_data_add_features()
# data_load_state.text('Done! (using cached data)')

# run model
model, _, _, _, _, _, _ = linear_regression_model_basic(df)
modelrf = random_forest_model(df)

# predict new dose
min_wt = float(df["weight"].min())
max_wt = float(df["weight"].max())
min_init = float(df["Initial Dose"].min())
max_init = float(df["Initial Dose"].max() )
min_tsh1 = float(df["TSH1"].min())
max_tsh1 = float(df["TSH1"].max() )

if widget=="text_box":
        weight = st.number_input("Enter your weight in Kg (48.0-140.0):", step=0.1, min_value=min_wt, max_value=max_wt)
        initial_weekly_dose = st.number_input("Enter your weekly dose (175.0-1300.0 mL):", step=25.0, min_value=min_init, max_value=max_init)
        TSH1 = st.number_input("Enter your original TSH level (1.0-13.0 mL):", step=1.0, min_value=min_tsh1, max_value=max_tsh1)
        health = st.number_input("On a scale of 1 (unwell) to 5 (well), how do you feel since taking the new medication dose?",
                           step=1, min_value=1, max_value=5)
elif widget=="slider":
        weight = st.slider("Enter your weight in Kg:", step=0.1, min_value=min_wt, max_value=max_wt)
        initial_weekly_dose = st.slider("Enter your weekly dose:", step=25.0, min_value=min_init, max_value=max_init)
        TSH1 = st.slider("Enter your original TSH level:", step=1.0, min_value=min_tsh1, max_value=max_tsh1)
        health = st.slider("On a scale of 1 (unwell) to 5 (well), how do you feel since taking the new medication dose?",
                           step=1, min_value=1, max_value=5)
else:
        print("No available widget choice has been made. Choose slider or text_box")

pregnant = st.radio("Are you currently pregnant?", ["no", "yes"], horizontal=True, index=0)

def get_final_dose(
        model,
        TSH1,
        weight,
        initial_weekly_dose,
        increase_increment = 25.0,
        unwell_cutoff = 2.0,
        health_cutoff = 3.0,
        low_tsh = 0.27,
        pregnant="no"
        ):
    def predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH_initial_high, TSH_initial_normal):  #, TSH2_const_val=2.0):
        new_data = {
            'weight': [weight],
            'Initial Dose': [initial_weekly_dose],
            'TSH1': [TSH1],
            'TSH_initial_high': [0],
            'TSH_initial_normal': [1],
            #'TSH_target':[2]
        }
        # Convert to DataFrame
        new_data_df = pd.DataFrame(new_data)

        # If you have a gender feature, encode it as was done during training
        # For example, if 'gender' was 'f' or 'm', and 'f' was encoded as 0
        if 'gender' in new_data_df.columns:
            new_data_df['gender'] = label_encoder.transform(['f'])

        # Make a prediction
        predicted_new_dose = model.predict(new_data_df)

        #return model.predict([[float(weight), float(initial_weekly_dose), float(TSH1), float(TSH2_const_val)]])[0]
        return predicted_new_dose[0]

    if pregnant=="yes":
        high_tsh = 2.5
    else:
        high_tsh = 4.2
    TSH_initial_high = int(TSH1 > high_tsh)
    TSH_initial_normal = int(low_tsh<=TSH1<=high_tsh)

    exact_dose_orig = np.round(predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH_initial_high, TSH_initial_normal), 1)
    exact_dose=exact_dose_orig
    if TSH1>=high_tsh and exact_dose_orig <= initial_weekly_dose:  # TSH is too high and new_dose is less than or equL to initial weekly dose
        exact_dose += increase_increment
    elif high_tsh >= TSH1 >= low_tsh:    # normal
        #if health == "well":
        #    new_dose = new_dose  # if feeling okay, leave it
        if health <=health_cutoff and TSH1 >= unwell_cutoff:
            exact_dose+=increase_increment  # if feeling unwell, increment
    return exact_dose, exact_dose_orig
    #new_dose = False
#increase_increment = 25.0
#unwell_cutoff = 2.0
#health_cutoff = 3.0
#low_tsh = 0.27
#if pregnant=="yes":
#    high_tsh = 2.5
#else:
#    high_tsh = 4.2
#TSH_initial_high = int(TSH1 > high_tsh)
#TSH_initial_normal = int(low_tsh<=TSH1<=high_tsh)
#
#exact_dose_orig = np.round(predict_new_dose(model, weight, initial_weekly_dose, TSH1, TSH_initial_high, TSH_initial_normal), 1)
#exact_dose=exact_dose_orig
#if TSH1>=high_tsh and exact_dose_orig <= initial_weekly_dose:  # TSH is too high and new_dose is less than or equL to initial weekly dose
#    exact_dose += increase_increment
#elif high_tsh >= TSH1 >= low_tsh:    # normal
#    #if health == "well":
#    #    new_dose = new_dose  # if feeling okay, leave it
#    if health <=health_cutoff and TSH1 >= unwell_cutoff:
#        exact_dose+=increase_increment  # if feeling unwell, increment

#new_dose = False
exact_dose = get_final_dose(model, TSH1, weight, initial_weekly_dose)
exact_dose_rf = get_final_dose(modelrf, TSH1, weight, initial_weekly_dose)
st.session_state["new_dose"] = math.floor(exact_dose / 25) * 25
st.session_state["new_dose_rf"] = math.floor(exact_dose_rf / 25) * 25
delta=round(st.session_state["new_dose"]-initial_weekly_dose, 1)
delta2=round(st.session_state["new_dose_rf"]-initial_weekly_dose, 1)

st.metric("Your new dose is:", value=f'{st.session_state["new_dose"]} or {st.session_state["new_dose_rf"]} (exact was {exact_dose_orig}) mL', delta=f'{delta} or {delta2} mL')
#st.metric("The original, exact predicted dose was:", exact_dose=f'{exact_dose} mL')
if health <= health_cutoff:
    st.write(f"Your dose was bumped up {increase_increment} mL as you stated that the current dose was making you feel unwell.")
