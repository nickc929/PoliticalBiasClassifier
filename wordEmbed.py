import sys

import numpy as np
from matplotlib import pyplot

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

# Fix random seed for reproducibility
np.random.seed(50)

# Sequences are front padded with zero vectors so all sequences have the same length
totalSequenceLength = 32
# wordVec dimension from GloVe (25, 50, 75, or 100)
wordVectorLength = 25

# Keeps track of the longest sequence
maxSequenceLength = 0

politicians = {
    # name : [party affiliation, pre election file path, post election file path]

    # DEMOCRATS
    "Amy Klobuchar": [
        0,
        'Processed Pre Election Democrats/amyKlobucharProcessed.txt',
        'Processed Post Election Democrats/amyKlobucharProcessed.txt'
    ],
    "Ben Cardin": [
        0,
        'Processed Pre Election Democrats/benCardinProcessed.txt',
        'Processed Post Election Democrats/benCardinProcessed.txt'
    ],
    "Bernie Sanders": [
        0,
        'Processed Pre Election Democrats/bernieSandersProcessed.txt',
        'Processed Post Election Democrats/bernieSandersProcessed.txt'
    ],
    "Bob Casey": [
        0,
        'Processed Pre Election Democrats/bobCaseyProcessed.txt',
        'Processed Post Election Democrats/bobCaseyProcessed.txt',
    ],
    "Bob Menendez": [
        0,
        'Processed Pre Election Democrats/bobMenendezProcessed.txt',
        'Processed Post Election Democrats/bobMenendezProcessed.txt',
    ],
    "Brian Schatz": [
        0,
        'Processed Pre Election Democrats/brianSchatzProcessed.txt',
        'Processed Post Election Democrats/brianSchatzProcessed.txt',
    ],
    "Chris Coons": [
        0,
        'Processed Pre Election Democrats/chrisCoonsProcessed.txt',
        'Processed Post Election Democrats/chrisCoonsProcessed.txt',
    ],
    "Chuck Schumer": [
        0,
        'Processed Pre Election Democrats/chuckSchumerProcessed.txt',
        'Processed Post Election Democrats/chuckSchumerProcessed.txt',
    ],
    "Cory Booker": [
        0,
        'Processed Pre Election Democrats/coryBookerProcessed.txt',
        'Processed Post Election Democrats/coryBookerProcessed.txt',
    ],
    "Dick Durbin": [
        0,
        'Processed Pre Election Democrats/dickDurbinProcessed.txt',
        'Processed Post Election Democrats/dickDurbinProcessed.txt',
    ],
    "Ed Markey": [
        0,
        'Processed Pre Election Democrats/edMarkeyProcessed.txt',
        'Processed Post Election Democrats/edMarkeyProcessed.txt',
    ],
    "Mark Warner": [
        0,
        'Processed Pre Election Democrats/markWarnerProcessed.txt',
        'Processed Post Election Democrats/markWarnerProcessed.txt',
    ],
    "Richard Blumenthal": [
        0,
        'Processed Pre Election Democrats/richardBlumenthalProcessed.txt',
        'Processed Post Election Democrats/richardBlumenthalProcessed.txt',
    ],
    "Sheldon Whitehouse": [
        0,
        'Processed Pre Election Democrats/sheldonWhitehouseProcessed.txt',
        'Processed Post Election Democrats/sheldonWhitehouseProcessed.txt',
    ],
    "Tim Kaine": [
        0,
        'Processed Pre Election Democrats/timKaineProcessed.txt',
        'Processed Post Election Democrats/timKaineProcessed.txt',
    ],

    # REPUBLICANS
    "Chuck Grassley": [
        1,
        'Processed Pre Election Republicans/chuckGrassleyProcessed.txt',
        'Processed Post Election Republicans/chuckGrassleyProcessed.txt',
    ],
    "Cory Gardner": [
        1,
        'Processed Pre Election Republicans/coryGardnerProcessed.txt',
        'Processed Post Election Republicans/coryGardnerProcessed.txt',
    ],
    "David Perdue": [
        1,
        'Processed Pre Election Republicans/davidPerdueProcessed.txt',
        'Processed Post Election Republicans/davidPerdueProcessed.txt',
    ],
    "John Cornyn": [
        1,
        'Processed Pre Election Republicans/johnCornynProcessed.txt',
        'Processed Post Election Republicans/johnCornynProcessed.txt',
    ],
    "Lindsey Graham": [
        1,
        'Processed Pre Election Republicans/lindseyGrahamProcessed.txt',
        'Processed Post Election Republicans/lindseyGrahamProcessed.txt',
    ],
    "Marco Rubio": [
        1,
        'Processed Pre Election Republicans/marcoRubioProcessed.txt',
        'Processed Post Election Republicans/marcoRubioProcessed.txt',
    ],
    "Mitt Romney": [
        1,
        'Processed Pre Election Republicans/mittRomneyProcessed.txt',
        'Processed Post Election Republicans/mittRomneyProcessed.txt',
    ],
    "Rand Paul": [
        1,
        'Processed Pre Election Republicans/randPaulProcessed.txt',
        'Processed Post Election Republicans/randPaulProcessed.txt',
    ],
    "Rick Scott": [
        1,
        'Processed Pre Election Republicans/rickScottProcessed.txt',
        'Processed Post Election Republicans/rickScottProcessed.txt',
    ],
    "Rob Portman": [
        1,
        'Processed Pre Election Republicans/robPortmanProcessed.txt',
        'Processed Post Election Republicans/robPortmanProcessed.txt',
    ],
    "Roy Blunt": [
        1,
        'Processed Pre Election Republicans/royBluntProcessed.txt',
        'Processed Post Election Republicans/royBluntProcessed.txt',
    ],
    "Shelley Moore Capito": [
        1,
        'Processed Pre Election Republicans/shelleyMooreCapitoProcessed.txt',
        'Processed Post Election Republicans/shelleyMooreCapitoProcessed.txt',
    ],
    "Ted Cruz": [
        1,
        'Processed Pre Election Republicans/tedCruzProcessed.txt',
        'Processed Post Election Republicans/tedCruzProcessed.txt',
    ],
    "Tim Scott": [
        1,
        'Processed Pre Election Republicans/timScottProcessed.txt',
        'Processed Post Election Republicans/timScottProcessed.txt',
    ],
    "Tom Cotton": [
        1,
        'Processed Pre Election Republicans/tomCottonProcessed.txt',
        'Processed Post Election Republicans/tomCottonProcessed.txt',
    ],
}


def sequence_embeddings(file_path):
    politician = []
    longest_seq_len = 0
    f = open(file_path, "r", encoding="utf-8")
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
                politician.append(np.concatenate((zeroesArray, tempEmbedArray)))
            else:
                print('WARNING: A sequence was skipped because it was too long (R)', line)
        if len(words) > longest_seq_len:
            longest_seq_len = len(words)
    f.close()

    return politician, longest_seq_len


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
demoEmbeds, demoMaxSeqLen = sequence_embeddings('mergedDemo.txt')

# Republican Embeddings
repubEmbeds, repubMaxSeqLen = sequence_embeddings('mergedRepub.txt')

print('Number of Republican Embeddings:', len(repubEmbeds))
print('Number of Democrat Embeddings:', len(demoEmbeds))


"""##############################################################"""


# Train/test split
repubSplit = int(0.8 * len(repubEmbeds))
demoSplit = int(0.8 * len(demoEmbeds))
X_train = np.array(repubEmbeds[:repubSplit] + demoEmbeds[:demoSplit])
y_train = np.array(repubSplit * [1] + demoSplit * [0])
X_test = np.array(repubEmbeds[repubSplit:] + demoEmbeds[demoSplit:])
y_test = np.array((len(repubEmbeds) - repubSplit) * [1] + (len(demoEmbeds) - demoSplit) * [0])
print('X_train', X_train.shape, 'X_test', X_test.shape)
print('y_train', y_train.shape, 'y_test', y_test.shape)


# Create model
model = Sequential()
model.add(LSTM(32, input_shape=(None, wordVectorLength)))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print(model.metrics_names)
print(model.metrics)


# Train model
epochs = 4
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=1)


# Test model
train_scores = model.evaluate(X_train, y_train, verbose=1)
predictions = model.predict(X_test)
y_pred = model.predict_classes(X_test)
scores = model.evaluate(X_test, y_test)
print('Epochs:', epochs)
print('train_scores:', train_scores)
print('test_scores: ', scores)
print('min, max, avg bias:', predictions.min(), predictions.max(), predictions.mean())

confusionMatrix = confusion_matrix(y_test, y_pred)
print('confusion matrix:\n', confusionMatrix)


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


# Individual politician scores
pre_election_scores = []
post_election_scores = []
pre_election_dem_scores = []
post_election_dem_scores = []
pre_election_rep_scores = []
post_election_rep_scores = []
for politician in politicians:
    pre_score = (model.predict(np.array(sequence_embeddings(politicians[politician][1])[0]))).mean()
    post_score = (model.predict(np.array(sequence_embeddings(politicians[politician][2])[0]))).mean()
    avg_score = 0
    # pre election
    politicians[politician].append(pre_score)
    pre_election_scores.append(pre_score)
    if politicians[politician][0] == 0:
        pre_election_dem_scores.append(pre_score)
    else:
        pre_election_rep_scores.append(pre_score)
    # post election
    politicians[politician].append(post_score)
    post_election_scores.append(post_score)
    if politicians[politician][0] == 0:
        post_election_dem_scores.append(post_score)
    else:
        post_election_rep_scores.append(post_score)
    print(politician, ':\t', pre_score, '\t', post_score, '\t', (pre_score+post_score) / 2)

pyplot.clf()
pyplot.scatter(pre_election_dem_scores, post_election_dem_scores, color='blue', marker='o', label='democrats')
pyplot.scatter(pre_election_rep_scores, post_election_rep_scores, color='red', marker='x', label='republicans')
pyplot.xlabel('pre-election')
pyplot.ylabel('post-election')
pyplot.axhline(predictions.mean())
pyplot.axvline(predictions.mean())
pyplot.plot([0.4, 0.6], [0.4, 0.6])
pyplot.legend(loc='upper left')
pyplot.show()


def most_biased_statements(file_path, mini, maxi):
    f = open(file_path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    return lines[mini], lines[maxi]


repub_predictions = model.predict(np.array(repubEmbeds))
# print('Indices', repub_predictions.argmin(), repub_predictions.argmax())
print('Republican Biases', repub_predictions.min(), repub_predictions.max())
print(most_biased_statements('mergedRepub.txt', repub_predictions.argmin(), repub_predictions.argmax()))

demo_predictions = model.predict(np.array(demoEmbeds))
# print('Indices', demo_predictions.argmin(), demo_predictions.argmax())
print('Democrat Biases', demo_predictions.min(), demo_predictions.max())
print(most_biased_statements('mergedDemo.txt', demo_predictions.argmin(), demo_predictions.argmax()))


# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, predictions)
area = auc(fpr, tpr)
pyplot.clf()
pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(area))
pyplot.xlabel('False positive rate')
pyplot.ylabel('True positive rate')
pyplot.title('ROC curve')
pyplot.legend(loc='best')
pyplot.show()