import numpy as np
import pandas as pd

import pygwalker as pyg

import streamlit as st
import streamlit.components.v1 as components

filepath = "../assets/loan_data.csv"
df = pd.read_csv(filepath)

st.title('Loan Borrower Data')
st.markdown("""
            <div style="text-align: justify;">
            The dataset contains information mainly about the borrower, such as the credit score and the status of payment. For a complete list of features, observe the following table.

            |    | Variable          | Description                                                                                                             |
            |---:|:------------------|:------------------------------------------------------------------------------------------------------------------------|
            |  0 | credit_policy     | 1 if the customer meets the credit underwriting criteria; 0 otherwise.                                                  |
            |  1 | purpose           | The purpose of the loan.                                                                                                |
            |  2 | int_rate          | The interest rate of the loan (more risky borrowers are assigned higher interest rates).                                |
            |  3 | installment       | The monthly installments owed by the borrower if the loan is funded.                                                    |
            |  4 | log_annual_inc    | The natural log of the self-reported annual income of the borrower.                                                     |
            |  5 | dti               | The debt-to-income ratio of the borrower (amount of debt divided by annual income).                                     |
            |  6 | fico              | The FICO credit score of the borrower.                                                                                  |
            |  7 | days_with_cr_line | The number of days the borrower has had a credit line.                                                                  |
            |  8 | revol_bal         | The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).                           |
            |  9 | revol_util        | The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available). |
            | 10 | inq_last_6mths    | The borrower's number of inquiries by creditors in the last 6 months.                                                   |
            | 11 | delinq_2yrs       | The number of times the borrower had been 30+ days past due on a payment in the past 2 years.                           |
            | 12 | pub_rec           | The borrower's number of derogatory public records.                                                                     |
            | 13 | not_fully_paid    | 1 if the loan is not fully paid; 0 otherwise. 
            </div>
            """, unsafe_allow_html=True
            )

st.dataframe(df, hide_index = True, use_container_width = True)
st.markdown("""
            <div style="text-align: justify;">
            The data has no null values and the features are set to the appropriate data type. Note that there is class imbalance as there are fewer examples of loans not fully paid. Specifically, eight thousand forty-five (8045) are fully paid while one thousand five hundred thirty-three (1533) are not. This is important to note since machine learning classifiers tend to underperform when class imbalance exists.
            </div>
            """, unsafe_allow_html=True
            )
st.write("## Properties")
st.write('###### Dimension:', df.shape)
st.write('###### Unique Course Types:', df['purpose'].sort_values().unique().tolist())

st.write('###### Summary Statistics')
st.dataframe(df.describe(), use_container_width = True)

st.write("## Visualization")

st.write("Visual explorations for this section are generated using [Pygwalker](https://docs.kanaries.net/pygwalker).")

pyg_html = pyg.walk(df, return_html=True)
 
# Embed the HTML into the Streamlit app
components.html(pyg_html, width=1000, height = 950, scrolling=True)
