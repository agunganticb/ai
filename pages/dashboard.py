import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("diabetes.csv")

st.title("📊 Dashboard Diabetes")

total = len(df)
positif = len(df[df["Outcome"] == 1])
negatif = len(df[df["Outcome"] == 0])

col1, col2, col3 = st.columns(3)

col1.metric("Total Data", total)
col2.metric("Positif Diabetes", positif)
col3.metric("Negatif Diabetes", negatif)

pie_data = pd.DataFrame({
    "Kategori": ["Positif", "Negatif"],
    "Jumlah": [positif, negatif]
})

fig = px.pie(
    pie_data,
    values="Jumlah",
    names="Kategori",
    title="Distribusi Diabetes"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Pasien")

st.dataframe(df.head(20))