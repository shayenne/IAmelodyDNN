# Import libraries and modules 
from librosa import *
import librosa.display
import scipy
import matplotlib.pyplot as plt
import numpy as np

# Preprocess all files in MedleyDB
import os

path = "../MedleyDB/Audio/"

#for filename in os.listdir(path):
#    print (filename)
#    example = "../MedleyDB/Audio/"+filename
    
    

# Converted to mono
y, sr = librosa.load(librosa.util.example_audio_file(), duration=5.0, mono=True) 
# Re-sampled to 16kHz
y_16k = librosa.resample(y, sr, 16000)
sr = 16000
print ("> Audio signal loaded...")


# Applying HPSS separation
# High frequency resolution - more clearly the frequencies
print ("> First HPSS decomposition (high-frequency resolution)...")
s = librosa.stft(y_16k, n_fft=4096, hop_length=int(4096*3/4), window=scipy.signal.hamming(4096))
h1, p1 = librosa.decompose.hpss(s)

# Getting first harmonic signal separated - need arguments of stft
h1_inverse = librosa.istft(h1, hop_length=int(4096*3/4), window=scipy.signal.hamming(4096))
librosa.output.write_wav(filename+"_H1.wav", h1_inverse, sr, norm=False)

# Getting first percussive signal separated - need arguments of stft
p1_inverse = librosa.istft(p1, hop_length=int(4096*3/4), window=scipy.signal.hamming(4096))
librosa.output.write_wav(filename+"_P1.wav", p1_inverse, sr, norm=False)


# P1 here has other frequency resolution
#print ("> Second HPSS decomposition (high-frequency resolution)...")
#p1 = librosa.stft(p1_inverse, n_fft=512, hop_length=int(512*3/4), window=scipy.signal.hamming(512))
#h2, p2 = librosa.decompose.hpss(p1_32)

# Getting second harmonic signal separated - need arguments of stft
#h2_inverse = librosa.istft(h2, hop_length=int(512*3/4), window=scipy.signal.hamming(512))
# Getting second precussive signal separated - need arguments of stft
#p2_inverse = librosa.istft(p2, hop_length=int(512*3/4), window=scipy.signal.hamming(512))