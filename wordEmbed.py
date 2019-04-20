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
#f = open(sys.argv[2], "r",  encoding="utf-8")
f = open('mergedDemo.txt', "r", encoding="utf-8")
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
#f = open(sys.argv[3], "r",  encoding="utf-8")
f = open('mergedRepub.txt', "r", encoding="utf-8")
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


"""##############################################################"""
# PRE ELECTION DEMOCRATS

# Amy Klobuchar
amyKPre = []
f = open('Processed Pre Election Democrats/amyKlobucharProcessed.txt', "r", encoding="utf-8")
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
            amyKPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Ben Cardin
benCPre = []
f = open('Processed Pre Election Democrats/benCardinProcessed.txt', "r", encoding="utf-8")
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
            benCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Bernie Sanders
bernieSPre = []
f = open('Processed Pre Election Democrats/bernieSandersProcessed.txt', "r", encoding="utf-8")
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
            bernieSPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Bob Casey
bobCPre = []
f = open('Processed Pre Election Democrats/bobCaseyProcessed.txt', "r", encoding="utf-8")
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
            bobCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Bob Menendez
bobMPre = []
f = open('Processed Pre Election Democrats/bobMenendezProcessed.txt', "r", encoding="utf-8")
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
            bobMPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Brian Schatz
brianSPre = []
f = open('Processed Pre Election Democrats/brianSchatzProcessed.txt', "r", encoding="utf-8")
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
            brianSPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Chris Coons
chrisCPre = []
f = open('Processed Pre Election Democrats/chrisCoonsProcessed.txt', "r", encoding="utf-8")
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
            chrisCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Chuck Schumer
chuckSPre = []
f = open('Processed Pre Election Democrats/chuckSchumerProcessed.txt', "r", encoding="utf-8")
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
            chuckSPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Cory Booker
coryBPre = []
f = open('Processed Pre Election Democrats/coryBookerProcessed.txt', "r", encoding="utf-8")
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
            coryBPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Dick Durbin
dickDPre = []
f = open('Processed Pre Election Democrats/dickDurbinProcessed.txt', "r", encoding="utf-8")
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
            dickDPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Ed Markey
edMPre = []
f = open('Processed Pre Election Democrats/edMarkeyProcessed.txt', "r", encoding="utf-8")
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
            edMPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Mark Warner
markWPre = []
f = open('Processed Pre Election Democrats/markWarnerProcessed.txt', "r", encoding="utf-8")
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
            markWPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Richard Blumenthal
richardBPre = []
f = open('Processed Pre Election Democrats/richardBlumenthalProcessed.txt', "r", encoding="utf-8")
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
            richardBPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Sheldon Whitehouse
sheldonWPre = []
f = open('Processed Pre Election Democrats/sheldonWhitehouseProcessed.txt', "r", encoding="utf-8")
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
            sheldonWPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Tim Kaine
timKPre = []
f = open('Processed Pre Election Democrats/timKaineProcessed.txt', "r", encoding="utf-8")
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
            timKPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

"""##############################################################"""
# POST ELECTION DEMOCRATS

# Amy Klobuchar
amyKPost = []
f = open('Processed Post Election Democrats/amyKlobucharProcessed.txt', "r", encoding="utf-8")
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
            amyKPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Ben Cardin
benCPost = []
f = open('Processed Post Election Democrats/benCardinProcessed.txt', "r", encoding="utf-8")
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
            benCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Bernie Sanders
bernieSPost = []
f = open('Processed Post Election Democrats/bernieSandersProcessed.txt', "r", encoding="utf-8")
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
            bernieSPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Bob Casey
bobCPost = []
f = open('Processed Post Election Democrats/bobCaseyProcessed.txt', "r", encoding="utf-8")
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
            bobCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Bob Menendez
bobMPost = []
f = open('Processed Post Election Democrats/bobMenendezProcessed.txt', "r", encoding="utf-8")
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
            bobMPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Brian Schatz
brianSPost = []
f = open('Processed Post Election Democrats/brianSchatzProcessed.txt', "r", encoding="utf-8")
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
            brianSPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Chris Coons
chrisCPost = []
f = open('Processed Post Election Democrats/chrisCoonsProcessed.txt', "r", encoding="utf-8")
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
            chrisCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Chuck Schumer
chuckSPost = []
f = open('Processed Post Election Democrats/chuckSchumerProcessed.txt', "r", encoding="utf-8")
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
            chuckSPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Cory Booker
coryBPost = []
f = open('Processed Post Election Democrats/coryBookerProcessed.txt', "r", encoding="utf-8")
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
            coryBPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Dick Durbin
dickDPost = []
f = open('Processed Post Election Democrats/dickDurbinProcessed.txt', "r", encoding="utf-8")
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
            dickDPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Ed Markey
edMPost = []
f = open('Processed Post Election Democrats/edMarkeyProcessed.txt', "r", encoding="utf-8")
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
            edMPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Mark Warner
markWPost = []
f = open('Processed Post Election Democrats/markWarnerProcessed.txt', "r", encoding="utf-8")
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
            markWPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Richard Blumenthal
richardBPost = []
f = open('Processed Post Election Democrats/richardBlumenthalProcessed.txt', "r", encoding="utf-8")
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
            richardBPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Sheldon Whitehouse
sheldonWPost = []
f = open('Processed Post Election Democrats/sheldonWhitehouseProcessed.txt', "r", encoding="utf-8")
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
            sheldonWPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Tim Kaine
timKPost = []
f = open('Processed Post Election Democrats/timKaineProcessed.txt', "r", encoding="utf-8")
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
            timKPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()


"""##############################################################"""
# PRE ELECTION REPUBLICANS

# Chuck Grassley
chuckGPre = []
f = open('Processed Pre Election Republicans/chuckGrassleyProcessed.txt', "r", encoding="utf-8")
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
            chuckGPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Cory Gardner
coryGPre = []
f = open('Processed Pre Election Republicans/coryGardnerProcessed.txt', "r", encoding="utf-8")
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
            coryGPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# David Perdue
davidPPre = []
f = open('Processed Pre Election Republicans/davidPerdueProcessed.txt', "r", encoding="utf-8")
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
            davidPPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# John Cornyn
johnCPre = []
f = open('Processed Pre Election Republicans/johnCornynProcessed.txt', "r", encoding="utf-8")
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
            johnCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Lindsey Graham
lindseyGPre = []
f = open('Processed Pre Election Republicans/lindseyGrahamProcessed.txt', "r", encoding="utf-8")
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
            lindseyGPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Marco Rubio
marcoRPre = []
f = open('Processed Pre Election Republicans/marcoRubioProcessed.txt', "r", encoding="utf-8")
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
            marcoRPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Mitt Romney
mittRPre = []
f = open('Processed Pre Election Republicans/mittRomneyProcessed.txt', "r", encoding="utf-8")
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
            mittRPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Rand Paul
randPPre = []
f = open('Processed Pre Election Republicans/randPaulProcessed.txt', "r", encoding="utf-8")
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
            randPPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Rick Scott
rickSPre = []
f = open('Processed Pre Election Republicans/rickScottProcessed.txt', "r", encoding="utf-8")
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
            rickSPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Rob Portman
robPPre = []
f = open('Processed Pre Election Republicans/robPortmanProcessed.txt', "r", encoding="utf-8")
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
            robPPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Roy Blunt
royBPre = []
f = open('Processed Pre Election Republicans/royBluntProcessed.txt', "r", encoding="utf-8")
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
            royBPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Shelley Moore Capito
shelleyMCPre = []
f = open('Processed Pre Election Republicans/shelleyMooreCapitoProcessed.txt', "r", encoding="utf-8")
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
            shelleyMCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Ted Cruz
tedCPre = []
f = open('Processed Pre Election Republicans/tedCruzProcessed.txt', "r", encoding="utf-8")
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
            tedCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Tim Scott
timSPre = []
f = open('Processed Pre Election Republicans/timScottProcessed.txt', "r", encoding="utf-8")
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
            timSPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Tom Cotton
tomCPre = []
f = open('Processed Pre Election Republicans/tomCottonProcessed.txt', "r", encoding="utf-8")
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
            tomCPre.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()


"""##############################################################"""
# POST ELECTION REPUBLICANS

# Chuck Grassley
chuckGPost = []
f = open('Processed Post Election Republicans/chuckGrassleyProcessed.txt', "r", encoding="utf-8")
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
            chuckGPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Cory Gardner
coryGPost = []
f = open('Processed Post Election Republicans/coryGardnerProcessed.txt', "r", encoding="utf-8")
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
            coryGPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# David Perdue
davidPPost = []
f = open('Processed Post Election Republicans/davidPerdueProcessed.txt', "r", encoding="utf-8")
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
            davidPPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# John Cornyn
johnCPost = []
f = open('Processed Post Election Republicans/johnCornynProcessed.txt', "r", encoding="utf-8")
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
            johnCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Lindsey Graham
lindseyGPost = []
f = open('Processed Post Election Republicans/lindseyGrahamProcessed.txt', "r", encoding="utf-8")
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
            lindseyGPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Marco Rubio
marcoRPost = []
f = open('Processed Post Election Republicans/marcoRubioProcessed.txt', "r", encoding="utf-8")
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
            marcoRPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Mitt Romney
mittRPost = []
f = open('Processed Post Election Republicans/mittRomneyProcessed.txt', "r", encoding="utf-8")
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
            mittRPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Rand Paul
randPPost = []
f = open('Processed Post Election Republicans/randPaulProcessed.txt', "r", encoding="utf-8")
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
            randPPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Rick Scott
rickSPost = []
f = open('Processed Post Election Republicans/rickScottProcessed.txt', "r", encoding="utf-8")
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
            rickSPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Rob Portman
robPPost = []
f = open('Processed Post Election Republicans/robPortmanProcessed.txt', "r", encoding="utf-8")
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
            robPPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Roy Blunt
royBPost = []
f = open('Processed Post Election Republicans/royBluntProcessed.txt', "r", encoding="utf-8")
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
            royBPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Shelley Moore Capito
shelleyMCPost = []
f = open('Processed Post Election Republicans/shelleyMooreCapitoProcessed.txt', "r", encoding="utf-8")
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
            shelleyMCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Ted Cruz
tedCPost = []
f = open('Processed Post Election Republicans/tedCruzProcessed.txt', "r", encoding="utf-8")
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
            tedCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Tim Scott
timSPost = []
f = open('Processed Post Election Republicans/timScottProcessed.txt', "r", encoding="utf-8")
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
            timSPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()

# Tom Cotton
tomCPost = []
f = open('Processed Post Election Republicans/tomCottonProcessed.txt', "r", encoding="utf-8")
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
            tomCPost.append(np.concatenate((zeroesArray, tempEmbedArray)))
        else:
            print('WARNING: A sequence was skipped because it was too long (R)', line)
    if len(words) > maxSequenceLength:
        maxSequenceLength = len(words)
f.close()


"""##############################################################"""


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
