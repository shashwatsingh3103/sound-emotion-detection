import streamlit as st
from st_audiorec import st_audiorec
import os
import numpy as np
import librosa
from keras.models import load_model

# Load the emotion detection model
model = load_model('yourmodel.h5')

def extract_features(path):
    x, sr = librosa.load(path, duration=8, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.array(features)
    reshaped_features = np.reshape(features, newshape=(1, 40, 1))
    predictions = model.predict(reshaped_features)
    predicted_label = np.argmax(predictions, axis=1)[0]

    emotion_categories = {
        0: "angry",
        1: "normal",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "pleasant",
        6: "sad",
    }

    # Decode the predicted label
    predicted_emotion = emotion_categories[predicted_label]
    return predicted_emotion

def save_wav_file(audio_data):
    # Save the audio data as a WAV file
    if not os.path.exists('sound'):
        os.makedirs('sound')
    file_path = os.path.join('sound', 'recorded_audio.wav')
    with open(file_path, 'wb') as f:
        f.write(audio_data)
    return file_path

def audiorec_demo_app():
    st.title('Emotion Detection')

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        file_path = save_wav_file(wav_audio_data)
        st.success(f'Audio file saved successfully at: {file_path}')
        predicted_emotion = predict_emotion(file_path)
        st.write(f'Predicted Emotion: {predicted_emotion}')

if __name__ == '__main__':
    audiorec_demo_app()
