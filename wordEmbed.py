import sys
import numpy as np
import string

# This file creates the word embeddings from Glove and Twitter Data

# Word Vector Dictionary
wordVectors = {}
f = open(sys.argv[1], "r",  encoding="utf-8")
for line in f:
    wordVectors[line.split()[0]] = line.split(' ', 1)[1].replace("\n", "")
f.close()

# Democratic Embeddings
demoEmbeds = []
f = open(sys.argv[2], "r",  encoding="utf-8")
for line in f:
    tempEmbed = []
    for word in line:
        if word in wordVectors:
            if "'s" in word:
                tempEmbed.append(wordVectors[word.split("'s")[0]])
                tempEmbed.append(wordVectors["'s"])
            else:
                tempEmbed.append(wordVectors[word])
    demoEmbeds.append(np.array(tempEmbed))
f.close()

# Republican Embeddings
repubEmbeds = []
f = open(sys.argv[3], "r",  encoding="utf-8")
for line in f:
    tempEmbed = []
    for word in line:
        if word in wordVectors:
            if "'s" in word:
                tempEmbed.append(wordVectors[word.split("'s")[0]])
                tempEmbed.append(wordVectors["'s"])
            else:
                tempEmbed.append(wordVectors[word])
    repubEmbeds.append(np.array(tempEmbed))
f.close()




