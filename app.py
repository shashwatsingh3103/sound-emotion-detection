import sounddevice as sd
import numpy as np
import wave
import os
import librosa
from keras.models import load_model
import streamlit as st

mo = load_model('yourmodel.h5')

def fe(path):
    x, sr = librosa.load(path, duration=8, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def pred(p):
    x = fe(p)
    x = np.array(x)
    l = np.reshape(x, newshape=(1, 40, 1))
    j = mo.predict(l)
    data = np.argmax(j, axis=1)

    category_names = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "pleasant",
        6: "sad",
    }

    # Decode each label
    decoded_labels = [category_names[i] for i in data]
    return decoded_labels

def main():
    st.title("Emotion Decoder ")

    # Load background and microphone images
    background_image = "background.jpg"
    microphone_image = "mic.jpg"
    
    st.image(microphone_image, width=100)  # Adjust width as needed

    # Add recording button
    if st.button("Start Recording"):
        record_audio()

    # Add "Record Again" button to refresh the website
    if st.button("Record Again"):
        st.experimental_rerun()

def record_audio():
    CHUNK = 1024
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5  # Adjust recording duration as needed
    OUTPUT_FILENAME = "recorded_audio.wav"

    frames = []

    with st.spinner('Recording in progress...'):
        recording = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=CHANNELS, dtype='int16')
        sd.wait()  # Wait until recording is finished
        frames.extend(recording)

    st.success("Recording stopped.")

    save_audio(frames, RATE, OUTPUT_FILENAME)

def save_audio(frames, rate, output_filename):
    folder_path = "backend_audio"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    wf = wave.open(os.path.join(folder_path, output_filename), "wb")
    wf.setnchannels(2)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(rate)
    wf.writeframes(frames.tobytes())
    wf.close()
    st.success("Audio recorded and saved successfully!")
    res = pred("backend_audio/recorded_audio.wav")
    st.write(res[0])

if __name__ == "__main__":
    main()
