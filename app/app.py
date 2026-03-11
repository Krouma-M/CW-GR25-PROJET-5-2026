import streamlit as st  # type: ignore
import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn import datasets # pyright: ignore[reportMissingModuleSource]

st.subheader("Bienvenue sur PediacPredicte")

st.sidebar.header("Les parametres")