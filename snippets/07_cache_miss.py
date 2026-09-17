import time

import numpy as np
import streamlit as st


@st.cache_data
def load_data(number):
    time.sleep(3)
    return f"Data loaded. This is {number}"


st.title("Caching Demo — key changes every time")

if st.button("Load Data"):
    st.write(load_data(np.random.randint(10)))
