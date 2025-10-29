import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import soundfile
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib


def record_audio(duration, freq):
    # Start recording
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1, dtype='float32')
    sd.wait()
    return recording

def save_audio(recording, filename, freq):
    # Convert the NumPy array to audio file
    write(filename, freq, recording)

# Sampling frequency
freq = 16000

# Recording duration
duration = 5

st.title("Audio Recorder")

# Allow the user to provide a file path to save the recorded audio
file_path = st.text_input("Enter the file path to save the recorded audio:", "recorded_audio.wav")

# Create a button to start recording
if st.button("Record"):
    st.info("Recording...")

    # Record audio
    recording = record_audio(duration, freq)

    # Remove the "Recording..." message
    st.empty()

    # Display the file uploader for saving the recorded audio
    #st.info("Recording finished!")
    
    # Save the recorded audio to the specified file path
    save_audio(recording, file_path, freq)
    
    st.success("Recording finished!")
    
    # Display the recorded audio
    st.audio(file_path, format="audio/wav")


def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        print(X)

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


features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
model = load_model("my_model_final_final.h5")

# Load the scaler from the file

# Load the trained encoder model
encoder = joblib.load("encoder.pkl")
#encoder = joblib.load(os.path.join(os.getcwd(), encoder_path))
scaler = joblib.load('scaler.pkl')
#scaler = joblib.load(os.path.join(os.getcwd(), encoder_path))

features = scaler.transform(features.reshape(1,-1))

predictions = model.predict(features)

#emotion_predictions = np.argmax(predictions, axis=1)

# Calculate percentage probabilities for each label
percentages = predictions * 100
percentages = percentages.tolist()

# Display the predictions on the Streamlit frontend
st.write("Predicted Emotions:")
probabilities = percentages[0]
emotions = encoder.categories_[0]

# Sort probabilities and emotions in descending order of probabilities
sorted_emotions_probs = sorted(zip(emotions, probabilities), key=lambda x: x[1], reverse=True)

# Display sorted probabilities and emotions
for emotion, probability in sorted_emotions_probs:
    st.write(f"{emotion}: {probability:.2f}%")


# Display the predictions on the Streamlit frontend
#st.write("Predicted Emotions:")
#for i, emotion in enumerate(encoder.categories_):
 #   st.write(f"{emotion}: {percentages[0][i]:.2f}%")


###emotion_predictions=encoder.inverse_transform(predictions)
#Display the predictions on the Streamlit frontend
#st.write("Predicted Emotions:")
#for emotion in emotion_predictions:
 #   st.write(emotion)
###