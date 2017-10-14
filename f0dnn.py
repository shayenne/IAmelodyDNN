# Create network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.initializers import RandomNormal
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
path_features =r'/var/tmp/IA/features/' # use your path
path_label =r'/var/tmp/IA/labels/' # use your path

list_mfcc = []
list_labels = []

# Loading files
for filename in os.listdir(path_mfcc):
    
    if filename.endswith("features.csv"):
        music = filename
        music = music[:-12]
        print (music)
        print("   Load", path_features+music+"features.csv")
        d1 = pd.read_csv(path_features+music+"features.csv",index_col=None, header=None)
        list_mfcc.append(d1)
        print("   Load", path_label+music+"labels.csv")
        d2 = pd.read_csv(path_label+music+"labels.csv",index_col=None, header=None)
        list_labels.append(d2)
        
# Grouping data 
frame = pd.concat(list_mfcc)
label = pd.concat(list_labels)


# Formating data
df2=pd.DataFrame.as_matrix(frame)

df4=pd.DataFrame.as_matrix(label)
df4 = df4[::2] # Hop size is doubled in features

df5 = np.zeros(shape=(df2.shape[0],1))
df5[:df4.shape[0],:] = df4

print("Data's shape:", df2.shape)
print("Data's shape:", df5.shape)


# Remove no vocal frames - train with only vocal frames
print("Removing no vocal frames from train data...")
no_vocal = np.where(df5[:,0] < 1)

for i in reversed(no_vocal):
    df2 = np.delete(df2, i, axis=0)
    df5 = np.delete(df5, i, axis=0)

print("Data's shape after remove:", df2.shape)
print("Data's shape after remove:", df5.shape)


# Train and test data
X = df2
Y = keras.utils.to_categorical(df5, num_classes=193) 
# To categorical formats the target to vectors of 0s and 1 on correspondent class


# Create model 
print("Create model...")
model = Sequential()
model.add(Dense(500, input_dim=513, activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(193, activation='softmax'))


# Compile model
print("Compile model...")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# The adam optimizer was used to reduce implementation details


# Fit the model
print("Starting model training...")
for i in range(100):
    model.fit(X, Y, 
              epochs=1, 
              batch_size=30, 
              validation_split=0.1, 
              shuffle=True)
    print("> Finished", i+1 , "epochs")
    # Serialize model to JSON
    model_json = model.to_json()
    with open("f0model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("f0model.h5")
    print("> Saved model to disk")


# Evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))