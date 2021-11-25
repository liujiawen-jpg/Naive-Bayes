import numpy as np
import re
token = re.compile('\\w*')

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱语句，0则相反
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #删除句子中重复的单词
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory)/np.float(numTrainDocs) #计算侮辱词语的概率
    p0Num = np.ones(numWords)
    p0Demon = numWords+1
    p1Num = np.ones(numWords)
    p1Demon = numWords+1
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num +=trainMatrix[i]
            p1Demon += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Demon+= np.sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Demon)
    p0Vect = np.log(p0Num/p0Demon)#为了防止太多太小的数相乘导致下溢出所以取对数处理
    return p0Vect,p1Vect,pAbusive


'''
vec2Classify:测试向量序列
p0Vec:第0类出现对应词的概率
p1Vec:第一类出现对应词的概率序列
p0Class:属于第0类先验概率
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, p1Class):
    p0 = np.sum(vec2Classify*p0Vec)+np.log(1.0-p1Class)
    p1 = np.sum(vec2Classify*p1Vec)+np.log(p1Class)
    if p0 > p1:
        return 0
    else:
        return 1

#便利函数用于封装一系列函数操作

def testingNB():
    dataList,testVec = loadDataSet()
    vocabList = createVocabList(dataList)
    trainMat = [setOfWords2Vec(vocabList,Doc) for Doc in dataList]
    p0v, p1v, pAb = trainNB0(trainMat, testVec)
    # testEntry = ['love', 'my', 'dalmation']
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print(classifyNB(thisDoc, p0v, p1v, pAb))


def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for words in inputSet:
        if words in vocabList:
            returnVec[vocabList.index(words)]+=1
    return returnVec


def textParse(bigString):
    listOfToken = token.findall(bigString)
    return [str.lower() for str in listOfToken if len(str)>0]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #给数据打上标签
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSPam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSPam)!= classList[docIndex]:
            errorCount+=1
    print("错误率是:",float(errorCount)/len(testSet))
    return float(errorCount)/len(testSet)

def airlineSentimentTest(dataSet):
    dataStr = np.array(dataSet['text']).tolist()
    docList = [textParse(sentence) for sentence in dataStr]
    classList = np.array(dataSet['sentiment']).tolist()
    vocabList = createVocabList(docList)
    testNum = int(len(classList)*0.3)
    testData = docList[:testNum]
    testClassList = classList[:testNum]
    trainData = docList[testNum:]
    trainClassList = classList[testNum:]
    trainMat = [bagOfWord2VecMN(vocabList,data) for data in trainData]
    # for data in trainData:
    #     trainMat.append(setOfWords2Vec(vocabList,data))
    p0V, p1V, pSPam = trainNB0(np.array(trainMat), np.array(trainClassList))
    errorCount = 0
    for i in range(len(testData)):
        wordVector = bagOfWord2VecMN(vocabList, testData[i])
        # wordVector = setOfWords2Vec(vocabList, testData[i])
        if classifyNB(np.array(wordVector), p0V, p1V, pSPam)!= testClassList[i]:
            errorCount+=1
    result = float(errorCount)/len(testData)
    print("错误率是: %.3f" % result)
    return result




