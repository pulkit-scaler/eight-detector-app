import streamlit as st

st.title("Session state")

# A plain variable: recreated on every rerun, so it never gets past 1.
naive_counter = 0

if "clicks" not in st.session_state:
    st.session_state.clicks = 0

if st.button("Increment"):
    naive_counter += 1
    st.session_state.clicks += 1

st.write(f"plain variable : {naive_counter}")
st.write(f"session state  : {st.session_state.clicks}")
