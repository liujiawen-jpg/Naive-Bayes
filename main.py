# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import pandas as pd
from bayes import *
import re
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
def to_index(sentiment):  # 写函数来转化
    return sentiment_to_index.get(sentiment)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # dataList,testVec = loadDataSet()
    # vocabList = createVocabList(dataList)
    # trainMat = [setOfWords2Vec(vocabList,Doc) for Doc in dataList]
    # print(trainMat)
    # p0v,p1v,pAb = trainNB0(trainMat,testVec)
    # print(p0v)
    # print_hi('PyCharm')
    # testingNB()

    sentiment_to_index = {'positive': 0, 'negative': 1}
    data = pd.read_csv('Tweets.csv')
    data = data[['airline_sentiment', 'text']]
    # data.head()
    data.airline_sentiment.value_counts()
    data_good = data[data.airline_sentiment == 'positive']
    data_negative = data[data.airline_sentiment == 'negative']
    # print(len(data_good))
    # print(len(data_negative))
    dataSet = pd.concat([data_good,data_negative])

    dataSet['sentiment'] = dataSet.airline_sentiment.apply(to_index)
    del dataSet['airline_sentiment']  # 删除原有的一列

    errorSum = 0.0
    for i in range(10):
        dataSet = dataSet.sample(len(dataSet))
        errorSum += airlineSentimentTest(dataSet)
    errorSum /= 10.0
    print("平均错误率是%.3f" % errorSum)
    sentiment_to_index = {'positive': 0,'negative': 1}


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
