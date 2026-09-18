import time

import numpy as np
import streamlit as st


@st.cache_data
def load_data():
    # The random draw is INSIDE the cached function, so it runs exactly once.
    # Every click after the first gets the stored return value back, number
    # included, which is why the number never changes.
    time.sleep(3)
    n = np.random.randint(10)
    return f"Data loaded. This is {n}"


st.title("Caching demo: stable key")

if st.button("Load Data"):
    st.write(load_data())
