import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


import streamlit as st
# from streamlit.hello.utils import show_code

def get_classifier(X, y, clf_name, smote, random_state):
    if smote == 'True':
        smt = SMOTE()
        X, y = smt.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    if clf_name == 'Extreme Gradient Boosted Trees':
        model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test) # Baseline
        y_pred_train = model.predict(X_train) # Baseline
        rec_train = recall_score(y_train, y_pred_train)
        rec_test = recall_score(y_test, y_pred)
    elif clf_name == 'TPOT-Optimized Pipeline':
        model = make_pipeline(
            MinMaxScaler(),
            GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.9000000000000001, min_samples_leaf=2, min_samples_split=5, n_estimators=100, subsample=0.9500000000000001)
            )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train) # Baseline
        rec_train = recall_score(y_train, y_pred_train)
        rec_test = recall_score(y_test, y_pred)
    return model, rec_train, rec_test

filepath = "loan_data.csv"
df = pd.read_csv(filepath)
y = df['not.fully.paid']
X_data = pd.get_dummies(df.drop(columns = ['not.fully.paid','int.rate'], axis = 1), drop_first = True)

random_state = st.sidebar.number_input('Random Seed', value = 42, step = 1)
clf = st.sidebar.selectbox("Classification Model", ('Extreme Gradient Boosted Trees', 'TPOT-Optimized Pipeline'))
smote = st.sidebar.selectbox("Synthetic Minority Oversampling Technique", ('True', 'False'))
model, rec_train, rec_test = get_classifier(X_data, y, clf, smote, random_state)
act = st.sidebar.radio("Action", ('Model Information', 'Make Prediction'))

def get_action(act_name, model):
    if act_name == 'Model Information':
        st.markdown(
            """ 
            # Trained Model Information
            """
            )
        st.write('Hyperparameters of the Selected Model', model.get_params())
        st.write('Recall Score on the Training Set:', rec_train)
        st.write('Recall Score on the Testing Set:', rec_test)
    else:
        st.write("### Input Features")
        cp= st.selectbox('Credit Policy', (1, 0))
        purp= st.selectbox('Purpose', ("all_other", "credit_card", "debt_consolidation", "educational", "home_improvement", "major_purchase", "small_business"))
        mi = st.number_input('Monthly Installment', format = '%f', value = 474.42)
        ai = st.number_input('Annual Income', format = '%f', value = 70000.0)
        dti = st.number_input('Debt-to-Income Ratio', format = '%f', value = 16.08)
        fico = st.number_input('FICO Credit Score', step = 1, value = 667)
        dcl = st.number_input('Days with Credit Line', format = '%f', value = 5429.958333)
        rb = st.number_input('Revolving Balance', format = '%f', value = 29797.0)
        rlur = st.number_input('Revolving Line Utilization Rate', format = '%f', value = 34.6)
        inq = st.number_input('Inquiries in the Last Six Months', step = 1, value = 3)
        delinq = st.number_input('Payment Delinquency in the Past Two Years', step = 2)
        derog = st.number_input('Number of Derogatory Public Records', step = 1)

        if purp == 'all_other':
            purp = np.zeros(6)
        elif purp == 'credit_card':
            purp = [1,0,0,0,0,0]
        elif purp == 'debt_consolidation':
            purp = [0,1,0,0,0,0]
        elif purp == 'educational':
            purp = [0,0,1,0,0,0]
        elif purp == 'home_improvement':
            purp = [0,0,0,1,0,0]
        elif purp == 'major_purchase':
            purp = [0,0,0,0,1,0]
        elif purp == 'small_business':
            purp = [0,0,0,0,0,1]
        input_feat = np.array(np.concatenate([[cp,mi,np.log(ai), dti, fico, dcl, rb, rlur, inq, delinq, derog],purp]).reshape(1,-1))
        if model.predict(input_feat)[0] == 1:
            st.write('#### THE LOAN WILL NOT BE FULLY PAID.')
        else:
            st.write('#### THE LOAN WILL BE FULLY PAID.')    
    return

get_action(act, model)