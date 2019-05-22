import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import word2vec
import jieba
import tensorflow as tf
import numpy as np
import time
from random import randint
from random import shuffle


'''
* ━━━━━━神兽出没━━━━━━
* 　　　┏┓　　　┏┓
* 　　┏┛┻━━━┛┻┓
* 　　┃　　　　　　　┃
* 　　┃　　　━　　　┃
* 　　┃　┳┛　┗┳　┃
* 　　┃　　　　　　　┃
* 　　┃　　　┻　　　┃
* 　　┃　　　　　　　┃
* 　　┗━┓　　　┏━┛Code is far away from bug with the animal protecting
* 　　　　┃　　　┃ 神兽保佑,代码无bug
* 　　　　┃　　　┃
* 　　　　┃　　　┗━━━┓
* 　　　　┃　　　　　　　┣┓
* 　　　　┃　　　　　　　┏┛
* 　　　　┗┓┓┏━┳┓┏┛
* 　　　　　┃┫┫　┃┫┫
* 　　　　　┗┻┛　┗┻┛
*
* ━━━━━━感觉萌萌哒━━━━━━
'''

#----------------------------------

def makeStopWord():
    with open('stopword.txt','r',encoding = 'utf-8') as f:
        lines = f.readlines()
    stopWord = []
    for line in lines:
        words = jieba.lcut(line,cut_all = False)
        for word in words:
            stopWord.append(word)
    return stopWord

def words2Array(lineList):
    linesArray=[]
    wordsArray=[]
    steps = []
    for line in lineList:
        t = 0
        p = 0
        for i in range(MAX_SIZE):
            if i<len(line):
                try:
                    wordsArray.append(model.wv.word_vec(line[i]))
                    p = p + 1
                except KeyError:
                    t=t+1
                    continue
            else:
               wordsArray.append(np.array([0.0]*dimsh))
        for i in range(t):
            wordsArray.append(np.array([0.0]*dimsh))
        steps.append(p)
        linesArray.append(wordsArray)
        wordsArray = []
    linesArray = np.array(linesArray)
    steps = np.array(steps)
    return linesArray, steps

def convert2Data(posArray, negArray, posStep, negStep):
    randIt = []
    data = []
    steps = []
    labels = []
    for i in range(len(posArray)):
        randIt.append([posArray[i], posStep[i], [1,0]])
    for i in range(len(negArray)):
        randIt.append([negArray[i], negStep[i], [0,1]])
    shuffle(randIt)
    for i in range(len(randIt)):
        data.append(randIt[i][0])
        steps.append(randIt[i][1])
        labels.append(randIt[i][2])
    data = np.array(data)
    steps = np.array(steps)
    return data, steps, labels

def getWords(file):
    wordList = []
    trans = []
    lineList = []
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        trans = jieba.lcut(line.replace('\n',''), cut_all = False)
        for word in trans:
            if word not in stopWord:
                wordList.append(word)
        lineList.append(wordList)
        wordList = []
    return lineList

def makeData(posPath,negPath):
    #获取词汇，返回类型为[[word1,word2...],[word1,word2...],...]
    pos = getWords(posPath)
    print("The positive data's length is :",len(pos))
    neg = getWords(negPath)
    print("The negative data's length is :",len(neg))
    #将评价数据转换为矩阵，返回类型为array
    posArray, posSteps = words2Array(pos)
    negArray, negSteps = words2Array(neg)
    #将积极数据和消极数据混合在一起打乱，制作数据集
    Data, Steps, Labels = convert2Data(posArray, negArray, posSteps, negSteps)
    return Data, Steps, Labels


#----------------------------------------------
# Word60.model   60维
# word2vec.model        200维

timeA=time.time()
word2vec_path = 'word2vec/word2vec.model'
model=gensim.models.Word2Vec.load(word2vec_path)
dimsh=model.vector_size
MAX_SIZE=25
stopWord = makeStopWord()

print("In train data:")
trainData, trainSteps, trainLabels = makeData('data/B/Pos-train.txt',
                                              'data/B/Neg-train.txt')
print("In test data:")
testData, testSteps, testLabels = makeData('data/B/Pos-test.txt',
                                           'data/B/Neg-test.txt')
trainLabels = np.array(trainLabels)

del model

print("-"*30)
print("The trainData's shape is:",trainData.shape)
print("The testData's shape is:",testData.shape)
print("The trainSteps's shape is:",trainSteps.shape)
print("The testSteps's shape is:",testSteps.shape)
print("The trainLabels's shape is:",trainLabels.shape)
print("The testLabels's shape is:",np.array(testLabels).shape)


num_nodes = 128
batch_size = 16
output_size = 2

graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,MAX_SIZE,dimsh))
    tf_train_steps = tf.placeholder(tf.int32,shape=(batch_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))

    tf_test_dataset = tf.constant(testData,tf.float32)
    tf_test_steps = tf.constant(testSteps,tf.int32)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = num_nodes,
                                             state_is_tuple=True)

    w1 = tf.Variable(tf.truncated_normal([num_nodes,num_nodes // 2], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([num_nodes // 2], stddev=0.1))

    w2 = tf.Variable(tf.truncated_normal([num_nodes // 2, 2], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1))
    
    def model(dataset, steps):
        outputs, last_states = tf.nn.dynamic_rnn(cell = lstm_cell,
                                                 dtype = tf.float32,
                                                 sequence_length = steps,
                                                 inputs = dataset)
        hidden = last_states[-1]

        hidden = tf.matmul(hidden, w1) + b1
        logits = tf.matmul(hidden, w2) + b2
        return logits
    train_logits = model(tf_train_dataset, tf_train_steps)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                logits=train_logits))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    test_prediction = tf.nn.softmax(model(tf_test_dataset, tf_test_steps))

num_steps = 20001
summary_frequency = 500


with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        offset = (step * batch_size) % (len(trainLabels)-batch_size)
        feed_dict={tf_train_dataset:trainData[offset:offset + batch_size],
                   tf_train_labels:trainLabels[offset:offset + batch_size],
                   tf_train_steps:trainSteps[offset:offset + batch_size]}
        _, l = session.run([optimizer,loss],
                           feed_dict = feed_dict)
        mean_loss += l
        if step >0 and step % summary_frequency == 0:
            mean_loss = mean_loss / summary_frequency
            print("The step is: %d"%(step))
            print("In train data,the loss is:%.4f"%(mean_loss))
            mean_loss = 0
            acrc = 0
            prediction = session.run(test_prediction)
            for i in range(len(prediction)):
                if prediction[i][testLabels[i].index(1)] > 0.5:
                    acrc = acrc + 1
            print("In test data,the accuracy is:%.2f%%"%((acrc/len(testLabels))*100))
#####################################
timeB=time.time()
print("time cost:",int(timeB-timeA))

