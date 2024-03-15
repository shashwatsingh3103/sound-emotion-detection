import os
import wave
import numpy as np
import pyaudio
import librosa
import streamlit as st
from keras.models import load_model

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
        1: "normal",
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
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5  # Adjust recording duration as needed
    OUTPUT_FILENAME = "recorded_audio.wav"

    audio = pyaudio.PyAudio()

    # Find and select the default input device
    input_device_index = None
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            input_device_index = i
            break

    if input_device_index is None:
        st.error("No input audio device found.")
        return

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=input_device_index)

    frames = []

    st.write("Recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Recording stopped.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    save_audio(frames, RATE, OUTPUT_FILENAME)

def save_audio(frames, rate, output_filename):
    folder_path = "backend_audio"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    wf = wave.open(os.path.join(folder_path, output_filename), 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    st.success("Audio recorded and saved successfully!")
    res = pred(os.path.join(folder_path, output_filename))
    st.write(res[0])

if __name__ == "__main__":
    main()
