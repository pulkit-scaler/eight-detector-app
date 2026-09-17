import numpy as np
import pandas as pd
import streamlit as st

st.header("User Inputs")
username = st.text_input("What is your name?", "Student")
age = st.slider("Select your age", 18, 100, 25)

if st.button("Say Hello"):
    st.success(f"Hello {username}! You are {age} years old.")

st.header("Random Data Plot")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)
