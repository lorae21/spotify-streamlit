import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import joblib

st.title("Song Popularity Prediction")
st.markdown('Model to predict the genre of a song given its features.\
            The genres that are determined here are:\
             edm - 0,\
            latin - 1,\
            pop - 2,\
            r&b - 3,\
            rap - 4,\
            rock - 5.')

st.header("Song Features")
# User input fields
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
key = st.selectbox("Key", range(12))
loudness = st.slider("Loudness", -60.0, 0.0, -30.0)

speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)

valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)

def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)

st.text('')
if st.button("Predict song genre"):
    result = predict(
        np.array([[ danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]]))
    st.text(result[0])
