# Import libraries and modules    
from librosa import *            # Manipulate and display audio files
import librosa.display
import scipy                     # Utilize signal functions
import numpy as np               # Manipulate arrays efficiently
import os, glob                  # Find files in directories
import csv                       # Manipulate .csv files

# All files from MedleyDB
path = "../MedleyDB/Audio/"

allFiles = glob.glob(path + "/*.wav")

for filename in os.listdir(path):
    if filename.endswith(".wav"):
        music = filename
        music = music[:-8]
        

    print ("--- Preprocessing...")
    print (filename)
    
    # Preprocess audio signal
    y, sr = librosa.load(path+filename, mono=True) # Converted to mono
    y_16k = librosa.resample(y, sr, 16000)
    sr = 16000

    print ("> Audio signal loaded...")

    # Applying HPSS separation
    # High frequency resolution - more clearly the frequencies
    print ("> First HPSS decomposition (high-frequency resolution)...")

    # STFT with Hamming window of 256ms (4096 samples) with overlap 0,75 (hop 0.25)
    s = librosa.stft(y_16k, n_fft=4096, hop_length=int(4096/4), window=scipy.signal.hamming(4096))
    h1, p1 = librosa.decompose.hpss(s)
    
    
    # P1 here has other frequency resolution
    print ("> Second HPSS decomposition (high-frequency resolution)...")
    # Getting signal - need arguments of stft
    p1_inverse = librosa.istft(p1, hop_length=int(4096/4), window=scipy.signal.hamming(4096))

    # STFT with Hamming window of 32ms (512 samples) with overlap 0,75 (hop 0.25)
    p1_32 = librosa.stft(p1_inverse, n_fft=512, hop_length=int(512/4), window=scipy.signal.hamming(512))
    h2, p2 = librosa.decompose.hpss(p1_32)
    
    
    """ Input for F0-Detection """    
    # Getting signal - need arguments of stft - Discarting frequencies?
    p1_toF0_inverse = librosa.istft(p1, hop_length=int(4096/4), window=scipy.signal.hamming(4096))

    # STFT with Hamming window of 64ms (1024 samples) with overlap 0,75 (hop 0.25)
    p1_toF0 = librosa.stft(p1_toF0_inverse, n_fft=1024, hop_length=int(1024/4), window=scipy.signal.hamming(1024))

    p1_toF0_log = librosa.amplitude_to_db(p1_toF0, ref=np.max)
    
    ## Rescale done for F0 Detection Model (Not cutted)
    p1_toF0_log = p1_toF0_log - p1_toF0_log.min()
    p1_toF0_log = p1_toF0_log / p1_toF0_log.max()  
    
    
    # Write features file 
    with open(path+'features/'+music+'_features.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        col = p1_toF0_log.shape[1]
        for i in range(col):
            spamwriter.writerow(p1_toF0_log[:,i])
        
        
    """ Input for VAD """    
    # The input for VAD is the whole signal s = h1 + h2 + p2 in MFCC features
    h1_mfcc = librosa.feature.melspectrogram(y=h1_inverse, sr=sr, n_fft=512, hop_length=128, n_mels=40, fmax=8000)
    h2_mfcc = librosa.feature.melspectrogram(y=h2_inverse, sr=sr, n_fft=512, hop_length=128, n_mels=40, fmax=8000)
    p2_mfcc = librosa.feature.melspectrogram(y=p2_inverse, sr=sr, n_fft=512, hop_length=128, n_mels=40, fmax=8000)
    
    # Write mfcc file 
    with open(path+'mfcc/'+music+'_mfcc.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        col = h1_mfcc.shape[1]
        row = np.ndarray(shape=(3*40,))
        for i in range(col):
            row[  :40] = h1_mfcc[:,i]
            row[40:80] = h2_mfcc[:,i]
            row[80:  ] = p2_mfcc[:,i]
            spamwriter.writerow(row) ### Put the mfcc from h1+h2+p2
        

    """ Labels """
    # Annotation 1 - Save with double frequency
    label_path = "../MedleyDB/Annotations/labels/"
    annot_path = "../MedleyDB/Annotations/"+music+"_MELODY1.csv"

    
    melody = []
    timestamps = []

    with open(annot_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            timestamps.append(float(row[0]))
            melody.append(float(row[1]))
        
        
    # Annotation sample rate is 44.1 kHz
    # Append element to make size equal of spectrogram
    melody.append(0)
    timestamps.append(0)
            
    # Convert annotation data to 16kHz
    Horig  = 256
    SRorig = 44100
    Hnew  = 128
    SRnew = 16000

    size = h1_mfcc.shape[1]

    j = np.arange(size) * (SRorig/Horig * Hnew/SRnew)   
    
    # Resample to 16kHz
    # None to values less equals 0
    melody_res = np.zeros(len(j))
    tmstamps = np.zeros(len(j))
    for i in range(len(j)):
        melody_res[i] = melody[int(j[i])]   # Get label more near from this frame resampled
        tmstamps[i] = timestamps[int(j[i])]
        if melody_res[i] <= 0:
            melody_res[i] = 0
            
            
    T = np.arange(193)
    # Value 4 is log2(1108.73) - log2(69.29)
    T1 = np.linspace(0,4, 193)
    
    # Define what value will be get from original annotation
    def find_nearest(array,value):
        i = (np.abs(array-value)).argmin()
        return i

    # Save labels file
    with open(label_path+music+'_labels.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(melody_res)):
            if melody_res[i] == 0:
                spamwriter.writerow("0")
            else:
                label = find_nearest(T1, np.log2(melody_res[i])-np.log2(69.29))
                spamwriter.writerow([label])       
            
"""  CODE TO SAVE AUDIO FILES          
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
print ("> Second HPSS decomposition (high-frequency resolution)...")
p1 = librosa.stft(p1_inverse, n_fft=512, hop_length=int(512*3/4), window=scipy.signal.hamming(512))
h2, p2 = librosa.decompose.hpss(p1)

# Getting second harmonic signal separated - need arguments of stft
h2_inverse = librosa.istft(h2, hop_length=int(512*3/4), window=scipy.signal.hamming(512))
librosa.output.write_wav(filename+"_H2.wav", h2_inverse, sr, norm=False)

# Getting second precussive signal separated - need arguments of stft
p2_inverse = librosa.istft(p2, hop_length=int(512*3/4), window=scipy.signal.hamming(512))
librosa.output.write_wav(filename+"_P2.wav", p2_inverse, sr, norm=False)

"""
