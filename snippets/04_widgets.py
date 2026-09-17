import streamlit as st

st.header("User Inputs")
username = st.text_input("What is your name?", "Student")
age = st.slider("Select your age", 18, 100, 25)

st.write(f"On this rerun, username={username!r} and age={age}")
