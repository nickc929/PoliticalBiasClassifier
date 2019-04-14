import sys

import numpy as np
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Fix random seed for reproducibility
np.random.seed(90123)

# Sequences are front padded with zero vectors so all sequences have the same length
totalSequenceLength = 36
# wordVec dimension from GloVe
wordVectorLength = 25

# Keeps track of the longest sequence
maxSequenceLength = 0

# This file creates the word embeddings from Glove and Twitter Data

# Word Vector Dictionary
wordVectors = {}
f = open(sys.argv[1], "r",  encoding="utf-8")
for line in f:
    tempList = []
    tempString = line.split(' ', 1)[1].replace("\n", "")
    for number in range(len(tempString.split())):
        tempList.append(float(tempString.split()[number]))
    wordVectors[line.split()[0]] = tempList
f.close()

# Democratic Embeddings
demoEmbeds = []
f = open(sys.argv[2], "r",  encoding="utf-8")
lines = f.readlines()
for line in lines:
    tempEmbed = []
    words = line.split()
    for word in words:
        if word in wordVectors:
            if "'s" in word:
                tempEmbed.append(wordVectors[word.split("'s")[0]])
                tempEmbed.append(wordVectors["'s"])
            else:
                tempEmbed.append(wordVectors[word])
    # demoEmbeds.append(np.array(tempEmbed))
    tempEmbedArray = np.array(tempEmbed)
    if tempEmbedArray.ndim == 2:
        if len(tempEmbed) <= totalSequenceLength:
            zeroesArray = np.zeros((totalSequenceLength - len(tempEmbed), wordVectorLength))
            demoEmbeds.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (D)', line)
    if len(tempEmbed) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Republican Embeddings
repubEmbeds = []
f = open(sys.argv[3], "r",  encoding="utf-8")
lines = f.readlines()
for line in lines:
    tempEmbed = []
    words = line.split()
    for word in words:
        if word in wordVectors:
            if "'s" in word:
                splitter = word.split("'s")
                if splitter[0]:
                    tempEmbed.append(wordVectors[word.split("'s")[0]])
                    tempEmbed.append(wordVectors["'s"])
            else:
                tempEmbed.append(wordVectors[word])
    # repubEmbeds.append(np.array(tempEmbed))
    tempEmbedArray = np.array(tempEmbed)
    if tempEmbedArray.ndim == 2:
        if len(tempEmbed) <= totalSequenceLength:
            zeroesArray = np.zeros((totalSequenceLength - len(tempEmbed), wordVectorLength))
            repubEmbeds.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

if maxSequenceLength > totalSequenceLength:
    print('maxSequenceLength', maxSequenceLength)

print('Number of repubEmbeds:', len(repubEmbeds))
print('Number of demEmbeds:', len(demoEmbeds))


# Train/test split
split_val = 800
X_train = np.array(repubEmbeds[:split_val] + demoEmbeds[:split_val])
y_train = np.array(split_val * [1] + split_val * [0])
X_test = np.array(repubEmbeds[split_val:] + demoEmbeds[split_val:])
y_test = np.array((len(repubEmbeds) - split_val) * [1] + (len(demoEmbeds) - split_val) * [0])
print('X_train', X_train.shape, 'X_test', X_test.shape)
print('y_train', y_train.shape, 'y_test', y_test.shape)


# Create model
model = Sequential()
model.add(LSTM(32, input_shape=(None, 25)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print(model.metrics_names)
print(model.metrics)

# Train model
epochs = 20
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)

# Test model
train_scores = model.evaluate(X_train, y_train)
scores = model.evaluate(X_test, y_test)
print('Epochs:', epochs)
print('train_scores:', train_scores)
print('test_scores: ', scores)


# Plots code from:
# https://machinelearningmastery.com/
# how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# Binary Cross-Entropy Loss Section

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
