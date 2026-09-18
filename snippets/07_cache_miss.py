import time

import numpy as np
import streamlit as st


@st.cache_data
def load_data(number):
    time.sleep(3)
    return f"Data loaded. This is {number}"


st.title("Caching demo: key changes every call")

if st.button("Load Data"):
    # The random draw is OUTSIDE, at the call site, so it is an argument and
    # therefore part of the cache key. A new key on every click means a new
    # entry, a fresh three second wait, and a different number.
    st.write(load_data(np.random.randint(10)))
