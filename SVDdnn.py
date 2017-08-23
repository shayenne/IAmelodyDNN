from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, add, Concatenate, merge, Add,Bidirectional, Activation, concatenate
from keras.models import model_from_json
import keras.layers
import keras
import numpy
import glob, os
import pandas as pd
np.random.seed(7)  # for reproducibility


# Read all files in path and concatenate them 
print("Loading features and labels for training...")
path_mfcc =r'/var/tmp/IA/mfcc/' # use your path
path_label =r'/var/tmp/IA/labels/' # use your path
#allFiles = glob.glob(path + "/*mfcc.csv")
list_mfcc = []
list_labels = []

i = 0
for filename in os.listdir(path_mfcc):
    
    if filename.endswith("mfcc.csv"):
        music = filename
        #music = "MusicDelta_Reggae_"
        music = music[:-8]
        print (music)
        print("   Load", path_mfcc+music+"mfcc.csv")
        d1 = pd.read_csv(path_mfcc+music+"mfcc.csv",index_col=None, header=None)
        list_mfcc.append(d1)
        print("   Load", path_label+music+"labels.csv")
        d2 = pd.read_csv(path_label+music+"labels.csv",index_col=None, header=None)
        list_labels.append(d2)
    i +=1
    if i == 5:
        break

frame = pd.concat(list_mfcc)
label = pd.concat(list_labels)


df2=pd.DataFrame.as_matrix(frame)
print (df2.shape)
# Trying to rescale the data
df2 = df2 - df2.mean()
df2 = df2 / df2.var()**2


df4=pd.DataFrame.as_matrix(label)
df4[df4 > 0] = 1
print (df4.shape)

print("Data's shape:", df2.shape)

# create data
x_train = df2
y_train = df4

# Important variables
max_features = 120
maxlen = 120
batch_size = 1000

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

# create model
print("Create model...")
model = Sequential()
model.add(Embedding(max_features, 50, input_length=maxlen))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Bidirectional(LSTM(50)))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile model
# 20 epochs without change
print("Compile model...")
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


# Fit the model
print("Starting model training...")
#callbacks = [
#    EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
#    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
#    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
#]

for i in range(1):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_split=0.1, shuffle=True) 
              #verbose=1, callbacks=callbacks)
    print("> Finished", i , "epochs")
        # serialize model to JSON
    model_json = model.to_json()
    with open("SVDmodel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("SVDmodel.h5")
    print("> Saved model to disk")
          
# evaluate the model          
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
