#!/bin/bash

# Update package list
apt-get update

# Install necessary build tools
apt-get install -y build-essential

# Download and extract PortAudio source code
wget http://www.portaudio.com/archives/pa_stable_v190600_20161030.tgz
tar -xzvf pa_stable_v190600_20161030.tgz
cd portaudio

# Configure, build, and install PortAudio
./configure && make && make install
