import numpy as np
import pandas as pd

import pygwalker as pyg

import streamlit as st
import streamlit.components.v1 as components

filepath = "loan_data.csv"
df = pd.read_csv(filepath)

# values = {"course_type": 'classroom', "year": 2011, "enrollment_count": 0, "pre_score": '0', "post_score": 0, "pre_requirement": 'None', "department": 'unknown'}
# df = df.fillna(value = values)
# df['pre_score'] = df['pre_score'].replace('-', '0')
# df['pre_score'] = df['pre_score'].astype(float)
# df['department'] = df['department'].str.strip().replace('Math', 'Mathematics')

st.title("Loan Borrower's Data")
st.markdown("""
            <div style="text-align: justify;">
            The dataset contains information mainly about the borrower, such as the credit score and the status of payment. For a complete list of features, refer to the table provided below.
            </div>
            <br>
            <table>
    <tr>
        <td>Variable</td>
        <td>Description</td>
    </tr>
    <tr>
        <td><code>credit_policy</code></td>
        <td>1 if the customer meets the credit underwriting criteria; 0 otherwise.</td>
    </tr>
    <tr>
        <td><code>purpose</code></td>
        <td>The purpose of the loan.</td>
    </tr>
    <tr>
        <td><code>int_rate</code></td>
        <td>The interest rate of the loan (more risky borrowers are assigned higher interest rates).</td>
    </tr>
    <tr>
        <td><code>installment</code></td>
        <td>The monthly installments owed by the borrower if the loan is funded.</td>
    </tr>
    <tr>
        <td><code>log_annual_inc</code></td>
        <td>The natural log of the self-reported annual income of the borrower.</td>
    </tr>
    <tr>
        <td><code>dti</code></td>
        <td>The debt-to-income ratio of the borrower (amount of debt divided by annual income).</td>
    </tr>
    <tr>
        <td><code>fico</code></td>
        <td>The FICO credit score of the borrower.</td>
    </tr>
    <tr>
        <td><code>days_with_cr_line</code></td>
        <td>The number of days the borrower has had a credit line.</td>
    </tr>
    <tr>
        <td><code>revol_bal</code></td>
        <td>The borrower&#39;s revolving balance (amount unpaid at the end of the credit card billing cycle).</td>
    </tr>
    <tr>
        <td><code>revol_util</code></td>
        <td>The borrower&#39;s revolving line utilization rate (the amount of the credit line used relative to total credit available).</td>
    </tr>
    <tr>
        <td><code>inq_last_6mths</code></td>
        <td>The borrower&#39;s number of inquiries by creditors in the last 6 months.</td>
    </tr>
    <tr>
        <td><code>delinq_2yrs</code></td>
        <td>The number of times the borrower had been 30+ days past due on a payment in the past 2 years.</td>
    </tr>
    <tr>
        <td><code>pub_rec</code></td>
        <td>The borrower&#39;s number of derogatory public records.</td>
    </tr>
</table>
            <br>
            The observations (or examples) are shown in the following dataframe. 
            """, unsafe_allow_html=True
            )
st.dataframe(df, hide_index = True, use_container_width = True)
st.markdown("""
            <div style="text-align: justify;">
            Note that the dataframe above is a cleaned version. The cleaning process include equating missing scores as 0 and labelling missing pre-requisites <code>None</code>, and replacing <code>Math</code> department with <code>Mathematics</code>.
            </div>
            """, unsafe_allow_html=True
            )
st.write("## Properties")
st.write('###### Dimension:', df.shape)
st.write('###### Unique Purposes:', df['purpose'].sort_values().unique().tolist())
st.write('###### Unique Inquiries:', df['inq.last.6mths'].sort_values().unique().tolist())
st.write('###### Payment Delinquency:', df['delinq.2yrs'].sort_values().unique().tolist())
st.write('###### Derogatory Public Records:', df['pub.rec'].sort_values().unique().tolist())
st.write('###### Summary Statistics')
st.dataframe(df.describe(), use_container_width = True)

st.write("## Visualization")

st.write("Visual explorations are possible with the help of [Pygwalker](https://docs.kanaries.net/pygwalker).")

pyg_html = pyg.walk(df, return_html=True)
 
# Embed the HTML into the Streamlit app
components.html(pyg_html, width=1000, height = 950, scrolling=True)