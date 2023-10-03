import numpy as np
import pandas as pd

import streamlit as st

from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
show_pages(
    [
        Page("main.py", "Introduction"),
        Page("pages/dataframe.py", "Borrower's Information for Loan Approval Assessment"),
        Page("pages/classification.py", "Classification Model"),
    ]
)

st.title("Machine Learning for Loan Approval: A Balancing Act Between Accuracy and Fairness")

st.markdown("""
            <h2> Background and Objective </h2>
            <div style="text-align: justify;">
            A start-up company wants to automate loan approvals by building a classifier to predict whether a loan will be paid back. In this situation, it is more important to accurately predict whether a loan will not be paid back rather than if a loan is paid back. Your manager will want to know how you accounted for this in training and evaluation your model. As a machine learning scientist, we need to build the classifier and prepare a report accessible to a broad audience.
            </div>
            <h2> Introduction </h2>
            <div style="text-align: justify;">
            The repayment rate of loans is influenced by a variety of factors, including the borrower's creditworthiness, the type of loan, and the economic climate. Studies found that borrowers with higher credit scores are more likely to repay their loans on time and in full. Furthermorele, student loans are typically repaid at a lower rate than other types of loans, such as personal loans or auto loansLastinally, the economic climate can also affect the loan repayment rate. During economic downturns, borrowers may be more likely to default on their loans due to job loss or other financial difficult of defaults.
            </div>
            """, unsafe_allow_html=True
            )
