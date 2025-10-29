import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import os

st.title("🎤 Emotion Recognition from Voice")

st.write("""
Record or upload your voice below — the app will analyze it and predict the emotion using a trained model.
""")

# ---- AUDIO INPUT SECTION ----
uploaded_audio = st.audio_input("Record your voice or upload an audio file")

if uploaded_audio is not None:
    # Save the uploaded/recorded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_audio.getbuffer())
        file_path = tmp_file.name

    st.audio(file_path, format="audio/wav")
    st.success("✅ Audio received successfully!")

    # ---- FEATURE EXTRACTION ----
    def extract_feature(file_name, **kwargs):
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")

        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma or contrast:
                stft = np.abs(librosa.stft(X))
            result = []
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                mfccs_std = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result.append(np.hstack((mfccs, mfccs_std)))
            if chroma:
                chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                chroma_std = np.std(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result.append(np.hstack((chroma_mean, chroma_std)))
            if mel:
                mel_mean = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                mel_std = np.std(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result.append(np.hstack((mel_mean, mel_std)))
            if contrast:
                contrast_mean = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                contrast_std = np.std(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                result.append(np.hstack((contrast_mean, contrast_std)))
            if tonnetz:
                tonnetz_mean = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                tonnetz_std = np.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                result.append(np.hstack((tonnetz_mean, tonnetz_std)))
        return np.concatenate(result)

    st.info("Extracting features from your audio...")
    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)

    # ---- MODEL + SCALER + ENCODER LOADING ----
    model = load_model("my_model_final_final(2).h5")
    encoder = joblib.load("encoder(2).pkl")
    scaler = joblib.load("scaler(2).pkl")

    # ---- PREDICTION ----
    features_scaled = scaler.transform(features.reshape(1, -1))
    predictions = model.predict(features_scaled)
    probabilities = (predictions * 100).tolist()[0]
    emotions = encoder.categories_[0]

    sorted_emotions_probs = sorted(zip(emotions, probabilities), key=lambda x: x[1], reverse=True)

    # ---- DISPLAY RESULTS ----
    st.subheader("🎯 Predicted Emotions")
    for emotion, probability in sorted_emotions_probs:
        st.write(f"**{emotion}:** {probability:.2f}%")

else:
    st.warning("Please record or upload an audio file to get started.")
