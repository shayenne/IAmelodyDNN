# Create network with Keras
from keras.preprocessing import sequence
from keras.initializers import RandomNormal
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Add, Bidirectional
from keras import optimizers
import keras.layers
import keras
import numpy as np
import pandas as pd
import glob
import os


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# Read all files in path and concatenate them 
print("Loading features and labels for training...")
path_mfcc =r'/var/tmp/IA/mfcc/' # use your path
path_label =r'/var/tmp/IA/labels/' # use your path

list_mfcc = []
list_labels = []

i = 0
for filename in os.listdir(path_mfcc):
    if i > 2:
        break
    if filename.endswith("mfcc.csv"):
        music = filename
        music = music[:-8]
        print (music)
        print("   Load", path_mfcc+music+"mfcc.csv")
        d1 = pd.read_csv(path_mfcc+music+"mfcc.csv",index_col=None, header=None)
        list_mfcc.append(d1)
        print("   Load", path_label+music+"labels.csv")
        d2 = pd.read_csv(path_label+music+"labels.csv",index_col=None, header=None)
        list_labels.append(d2)
    i +=1
    
# Grouping data 
frame = pd.concat(list_mfcc)
label = pd.concat(list_labels)


# Formating data
df2=pd.DataFrame.as_matrix(frame)
df4=pd.DataFrame.as_matrix(label)
df4[df4 > 0] = 1
print("Data's shape:", df2.shape)
print("Data's shape:", df4.shape)


# Rescale the data over all training set
meanDF = df2 - np.mean(df2, axis=0).reshape(1, df2.shape[1])
stdDF  = meanDF / np.std(meanDF, axis=0).reshape(1, df2.shape[1])
print("Data's shape:", df2.shape)


# Train and test data
x_train = df2
y_train = df4

# Padding sequences for input 
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)


# Important variables
max_features = 120
hidden_neurons = 50
maxlen = 120


# Create model
print("Create model...")
model = Sequential()
model.add(Embedding(max_features, 50, input_length=maxlen))
model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_initializer = RandomNormal(mean = 0, stddev = 0.1))))
model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_initializer = RandomNormal(mean = 0, stddev = 0.1))))
model.add(Bidirectional(LSTM(50, kernel_initializer = RandomNormal(mean = 0, stddev = 0.1))))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compile model
print("Compile model...")
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
# The adam optimizer was used to reduce implementation details


# Fit the model
print("Starting model training...")
for i in range(25):
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=1,
              validation_split=0.1, shuffle=True) 
    print("> Finished", i+1 , "epochs")
    # Serialize model to JSON
    model_json = model.to_json()
    with open("SVDmodel.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("SVDmodel.h5")
    print("> Saved model to disk")
          
        
# Evaluate the model          
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
