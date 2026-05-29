import streamlit as st
import pandas as pd

df = pd.read_csv("diabetes.csv")

st.title("📋 Monitoring Data Pasien")

st.dataframe(df)