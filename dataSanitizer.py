import sys
import string

def main():

    fileName = sys.argv[1]
    inFile = open(fileName, "r")
    # Parse words into list, split by words, and declare other variables
    corpus = inFile.readlines()
    dataList = []
    targetList = []
    outFile = outFile = open("processedPost2016ElectionRepublicans.txt","w")
    for line in corpus:
        line = line.replace("\\xe2\\x80\\x99","'")
        index = line.index("\tb") - 1
        newLine = line[2:index]
        if "http" in newLine:
            index = line.index("http") - 1
            newLine = line[2:index]
        
        print(newLine, file=outFile)
        
        if (line[-2] == '0'):
            targetList.append(0)
        else:
            targetList.append(1)


if __name__ == "__main__":
    main()
