import numpy as np
import random
import pickle
import re
import pandas as pd
empty_vector = []
for i in range(0, 100):
    empty_vector.append(float(0.0))
onevector = []
for i in range(0, 10):
    onevector.append(float(1))
zerovector = []
for i in range(0, 10):
    zerovector.append(float(0))
train_path1="E:/tianchi_tran/my_cnn/competation_data/cikm_english_train_20180516.txt"
train_path2="E:/tianchi_tran/my_cnn/competation_data/cikm_spanish_train_20180516.txt"
train_path3="E:/tianchi_tran/my_cnn/competation_data/cikm_test_a_20180516.txt"
def build_vocab():
    """
    convert question's words into correspond to index about the train and test data
    :return:
    """
    code = int(0)
    vocab = {} #words correspond it's index about train data and test data for question
    vocab['UNKNOWN'] = code #词语对应的下标
    code += 1
    train_x1=[]
    train_x2=[]
    train_y=[]
    max_len=60
    for line in open(train_path1,encoding='utf-8'):
        english_1,span_1,english_2,span_2,lable = line.strip().split("\t")
        # items = line.strip().split('$$$$$') #get each train
        # for i in range(2, 4):
        words1= span_1.split()#得到输入问句<a>填充之后的句子
        words2=span_2.split()
        if len(words1)>max_len:
            max_len=len(words1)
        if len(words2)>max_len:
            max_len=len(words2)
        train_x1.append(words1)
        train_x2.append(words2)
        for word in words1:
            if not word in vocab:
                vocab[word] = code
                code += 1
        for word in words2:
            if not word in vocab:
                vocab[word] = code
                code += 1
        for tem_labe in lable:
            train_y.append(int(tem_labe))
    for line in open(train_path2,encoding='utf-8'):
        span_1, english_1,  span_2, english_2,lable = line.strip().split("\t")
        # items = line.strip().split('$$$$$') #get each train
        # for i in range(2, 4):
        words1 = span_1.split()  # 得到输入问句<a>填充之后的句子
        words2 = span_2.split()
        if len(words1) > max_len:
            max_len = len(words1)
        if len(words2) > max_len:
            max_len = len(words2)
        train_x1.append(words1)
        train_x2.append(words2)
        for word in words1:
            if not word in vocab:
                vocab[word] = code
                code += 1
        for word in words2:
            if not word in vocab:
                vocab[word] = code
                code += 1
        for tem_labe in lable:
            train_y.append(int(tem_labe))
    print(max_len)
    return vocab,train_y,train_x1,train_x2
def build_vocab_test():
    """
    convert question's words into correspond to index about the train and test data
    :return:
    """
    code = int(0)
    vocab = {} #words correspond it's index about train data and test data for question
    vocab['UNKNOWN'] = code #词语对应的下标
    code += 1
    test_x1=[]
    test_x2=[]
    max_len=60
    for line in open(train_path3,encoding='utf-8'):
        span_1,span_2= line.strip().split("\t")

        # items = line.strip().split('$$$$$') #get each train
        # for i in range(2, 4):
        words1 = span_1.split()  # 得到输入问句<a>填充之后的句子
        words2 = span_2.split()
        if len(words1) > max_len:
            max_len = len(words1)
        if len(words2) > max_len:
            max_len = len(words2)
        test_x1.append(words1)
        test_x2.append(words2)
        for word in words1:
            if not word in vocab:
                vocab[word] = code
                code += 1
        for word in words2:
            if not word in vocab:
                vocab[word] = code
                code += 1
    return test_x1,test_x2
def rand_qa(qalist):
    #随机选择答案
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]

def read_alist():
    """
    get the question'answer
    """
    alist = []
    for line in open(train_path1):
        items = line.strip().split("$$$$$")
        alist.append(items[3])
    print('read_alist done ......')
    return alist

def vocab_plus_overlap(vectors, sent, over, size):
    global onevector
    global zerovector
    oldict = {}
    words = over.split('_')
    if len(words) < size:
        size = len(words)
    for i in range(0, size):
        if words[i] == '<a>':
            continue
        oldict[words[i]] = '#'
    matrix = []
    words = sent.split('_')
    if len(words) < size:
        size = len(words)
    for i in range(0, size):
        vec = read_vector(vectors, words[i])
        newvec = vec.copy()
        #if words[i] in oldict:
        #    newvec += onevector
        #else:
        #    newvec += zerovector
        matrix.append(newvec)
    return matrix

def load_vectors():
    vectors = {}
    for line in open('/home/admin/chen/chen/cnn_qa/insuranceQA-cnn-lstm-master/insuranceQA/vectors.nobin'):# each word'vector
        items = line.strip().split(' ')
        if (len(items) < 401):
            continue
        vec = []
        for i in range(1, 401):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    return vectors

def read_vector(vectors, word):
    global empty_vector
    if word in vectors:
        return vectors[word]
    else:
        return empty_vector
        #return vectors['</s>']

def load_test_and_vectors():
    # testList = []
    # for line in open('/home/admin/chen/chen/cnn_qa/insuranceQA-cnn-lstm-master/insuranceQA/test1.sample'):
    #     testList.append(line.strip())
    vectors = load_vectors()
    return vectors

def load_train_and_vectors():
    trainList = []
    for line in open(train_path):
        trainList.append(line.strip())
    vectors = load_vectors()
    return trainList, vectors

def load_data_val_10(testList, vectors, index):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = testList[index].split(' ')
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def read_raw():
    """
    :return:get train set by raw format
    """
    raw = []
    for line in open(train_path):
        items = line.strip().split('$$$$$')
        if items[0] == '1':
            raw.append(items)
    return raw

def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, 263): #这个地方有问题，本身的长度答案可能大于51，所以应该使用最大长度
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def is_float(s):
    s = str(s)
    if s.count('.') == 1:  # 判断小数点个数
        sl = s.split('.')  # 按照小数点进行分割
        left = sl[0]  # 小数点前面的
        right = sl[1]  # 小数点后面的
        if left.startswith('-') and left.count('-') == 1 and right.isdigit():
            lleft = left.split('-')[1]  # 按照-分割，然后取负号后面的数字
            if lleft.isdigit():
                return True
        elif left.isdigit() and right.isdigit():
            # 判断是否为正小数
            return True
    return False
def load_word_vec():
    vec_path="E:/tianchi_tran/my_cnn/competation_data/wiki.es.vec"

    words_vec={}
    with open(vec_path,'r',encoding='utf-8') as file:
        lines=file.readlines()
        k=0
        for line in lines:
            if k==0:
                print(line)
                k+=1
                continue
            wordvec=line.split(" ")
            if len(wordvec[:-1])!=301:
                print(len(wordvec))
                print(line)
            word=wordvec[0]
            vec=[float(ele) for ele in wordvec[1:-1]]
            words_vec[word]=vec
            if len(vec)!=300:
                print(line)
                print(len(vec))
            k+=1
    vec = np.random.rand(300) - 0.5
    words_vec['ttttt']=vec
    f = open('E:/tianchi_tran/my_cnn/competation_data/spanish_vec.pkl', 'wb')
    pickle.dump(words_vec, f)
    f.close()
def distance_str(str1,str2):
    dp=np.zeros((len(str1)+1,len(str2)+1))
    m=len(str1)
    n=len(str2)
    for k in range(1,m+1):
        dp[k][0]=k
    for k in range(1,n+1):
        dp[0][k]=k
    for k in range(1,m+1):
        for j in range(1,n+1):
            dp[k][j]=min(dp[k-1][j],dp[k][j-1])+1 #这里表示上边和下边的数值最小数值
            if str1[k-1]==str2[j-1]:
                dp[k][j]=min(dp[k][j],dp[k-1][j-1])
            else:
                dp[k][j]=min(dp[k][j],dp[k-1][j-1]+1)
    return dp[-1][-1]

def load_data_temp(words_vec,train_x1,train_x2,train_y, size):
    """
    :param vocab: word and index
    :param alist: question'answer
    :param raw:   original question and answer
    :param size: batch size=30
    :return:  x_train_1 表示对应的问题的下标
              x_train_2表示对应的答案的下标
              x_train_3随机选择的答案下标
    """
    x_train_1 = []  #转变成词语对应的下标
    x_train_2 = []
    y_train3=[]
    real_sent={}
    all_words=list(words_vec.keys())
    for i in range(0, size):
        train_oen_x1=[] #当前句子向量,最终其长度均为70
        train_oen_x2=[] #
        train_index=random.randint(0, len(train_y) - 1)
        temp_train_y = train_y[train_index] #随机选择训练集
        temp_train_x1=train_x1[train_index]
        temp_train_x2=train_x2[train_index]
        for one in temp_train_x1:#对于每一个句子的没一个单词转化为词向量
            try:
                # one=one.replace("¿","").replace("?","").replace(",","").replace(".","").replace(";","").replace("!","").replace("!","").replace(":","")\
                #     .replace('"','').replace('(','').replace(')','').replace("'",'').lower()
                ones = ''.join(re.findall(r'\w',one)).lower()
                train_oen_x1.append(np.array(words_vec[ones]))
            except:
                # print("444444444")
                # print(one)
                vec=np.random.rand(300) - 0.5
                words_vec[ones]=vec
                train_oen_x1.append(vec)
        for one in temp_train_x2:  # 对于每一个句子的没一个单词转化为词向量
            try:
                # one = one.replace("¿", "").replace("?", "").replace(",", "").replace(".", "").replace(";","").replace("!","").replace("!","").replace(":","")\
                #     .replace('"','').replace('(','').replace(')','').replace("'",'').lower()
                ones=''.join(re.findall(r'\w', one)).lower()
                train_oen_x2.append(np.array(words_vec[ones]))
            except:
                # print("444444444")
                # print(one)
                vec = np.random.rand(300) - 0.5
                words_vec[ones] = vec
                train_oen_x2.append(vec)
        rest_len_1=70-len(train_oen_x1)
        rest_len_2 = 70 - len(train_oen_x2)
        for k in range(rest_len_1):
            train_oen_x1.append(np.array(words_vec['ttttt']))
        for k in range(rest_len_2):
            train_oen_x2.append(np.array(words_vec['ttttt']))

        x_train_1.append(train_oen_x1)
        x_train_2.append(train_oen_x2)
        y_train3.append(temp_train_y)
    return np.array(x_train_1),np.array(x_train_2),np.array(y_train3),words_vec
def load_data_temp_test(words_vec,test_x1,test_x2,start_index):
    """
    :param vocab: word and index
    :param alist: question'answer
    :param raw:   original question and answer
    :param size: batch size=30
    :return:  x_train_1 表示对应的问题的下标
              x_train_2表示对应的答案的下标
              x_train_3随机选择的答案下标
    """
    x_test_1 = []  #转变成词语对应的下标
    x_test_2 = []
    # f = open('/home/admin/chen/chen/tianchi_trans/translate_data/spanish_vec.pkl', 'rb')
    # words_vec = pickle.load(f)
    all_words=list(words_vec.keys())
    for test_index in range(start_index, start_index+10):
        test_oen_x1=[] #当前句子向量,最终其长度均为70
        test_oen_x2=[] #

        temp_test_x1=test_x1[test_index]
        temp_test_x2=test_x2[test_index]
        for one in temp_test_x1:#对于每一个句子的没一个单词转化为词向量
            try:
                # one=one.replace("¿","").replace("?","").replace(",","").replace(".","").replace(";","").replace("!","").replace("!","").replace(":","")\
                #     .replace('"','').replace('(','').replace(')','').replace("'",'').lower()
                ones = ''.join(re.findall(r'\w',one)).lower()
                test_oen_x1.append(np.array(words_vec[ones]))
            except:
                # print("444444444")
                # print(one)
                vec=np.random.rand(300) - 0.5
                words_vec[ones]=vec
                test_oen_x1.append(vec)
        for one in temp_test_x2:  # 对于每一个句子的没一个单词转化为词向量
            try:
                # one = one.replace("¿", "").replace("?", "").replace(",", "").replace(".", "").replace(";","").replace("!","").replace("!","").replace(":","")\
                #     .replace('"','').replace('(','').replace(')','').replace("'",'').lower()
                ones=''.join(re.findall(r'\w', one)).lower()
                test_oen_x2.append(np.array(words_vec[ones]))
            except:
                # print("444444444")
                # print(one)
                vec = np.random.rand(300) - 0.5
                words_vec[ones] = vec
                test_oen_x2.append(vec)
        rest_len_1=70-len(test_oen_x1)
        rest_len_2 = 70 - len(test_oen_x2)
        for k in range(rest_len_1):
            test_oen_x1.append(np.array(words_vec['ttttt']))
        for k in range(rest_len_2):
            test_oen_x2.append(np.array(words_vec['ttttt']))

        x_test_1.append(test_oen_x1)
        x_test_2.append(test_oen_x2)
    return np.array(x_test_1),np.array(x_test_2),words_vec
def load_data_6(train_x1,train_x2,train_y, size):
    """
    :param vocab: word and index
    :param alist: question'answer
    :param raw:   original question and answer
    :param size: batch size=30
    :return:  x_train_1 表示对应的问题的下标
              x_train_2表示对应的答案的下标
              x_train_3随机选择的答案下标
    """
    x_train_1 = []  #转变成词语对应的下标
    x_train_2 = []
    y_train3=[]
    f = open('/home/admin/chen/chen/tianchi_trans/translate_data/spanish_vec.pkl', 'rb')
    words_vec = pickle.load(f)
    real_sent={}
    all_words=list(words_vec.keys())
    for i in range(0, size):
        train_oen_x1=[] #当前句子向量
        train_oen_x2=[] #
        train_index=random.randint(0, len(train_y) - 1)
        temp_train_y = train_y[train_index] #随机选择训练集
        temp_train_x1=train_x1[train_index]
        temp_train_x2=train_x2[train_index]
        for one in temp_train_x1:#对于每一个句子的没一个单词转化为词向量
            try:
                # one=one.replace("¿","").replace("?","").replace(",","").replace(".","").replace(";","").replace("!","").replace("!","").replace(":","")\
                #     .replace('"','').replace('(','').replace(')','').replace("'",'').lower()
                ones = ''.join(re.findall(r'\w',one)).lower()
                train_oen_x1.append(words_vec[ones])
            except:
                if ones.isdigit():
                    print("444444444")
                    print(one)
                    print(ones)
                    vec=np.random.rand(300) - 0.5
                    words_vec[ones]=vec
                    train_oen_x1.append(vec)
                else:
                    flag=False
                    for word_temp in all_words:
                        if distance_str(word_temp,ones)<=1:
                            flag=True
                            words_vec[ones]=words_vec[word_temp]
                            train_oen_x1.append(words_vec[word_temp])
                            break
                    if flag==False:
                        print("555555555555")
                        print(one)
                        print(ones)
                        vec = np.random.rand(300) - 0.5
                        words_vec[ones] = vec
                        train_oen_x1.append(vec)
        for one in temp_train_x2:  # 对于每一个句子的没一个单词转化为词向量
            try:
                # one = one.replace("¿", "").replace("?", "").replace(",", "").replace(".", "").replace(";","").replace("!","").replace("!","").replace(":","")\
                #     .replace('"','').replace('(','').replace(')','').replace("'",'').lower()
                ones=''.join(re.findall(r'\w', one)).lower()
                train_oen_x2.append(words_vec[ones])
            except:
                flag = False
                if ones.isdigit():
                    print("444444444")
                    print(one)
                    print(ones)
                    vec = np.random.rand(300) - 0.5
                    words_vec[ones] = vec
                    train_oen_x2.append(vec)
                else:
                    for word_temp in all_words:
                        if distance_str(word_temp, ones) <= 1:
                            flag = True
                            words_vec[ones] = words_vec[word_temp]
                            train_oen_x2.append(words_vec[word_temp])
                            break
                    if flag == False:
                        print("555555555555")
                        print(one)
                        print(ones)
                        vec = np.random.rand(300) - 0.5
                        words_vec[ones] = vec
                        train_oen_x2.append(vec)
        rest_len_1 = 70 - len(train_oen_x1)
        rest_len_2 = 70 - len(train_oen_x2)
        for k in range(rest_len_1):
            train_oen_x1.append(words_vec['ttttt'])
        for k in range(rest_len_2):
            train_oen_x2.append(words_vec['ttttt'])
        x_train_1.append(train_oen_x1)
        x_train_2.append(train_oen_x2)
        y_train3.append(temp_train_y)
    f = open('/home/admin/chen/chen/tianchi_trans/translate_data/train_data.pkl', 'wb')
    pickle.dump(np.array(x_train_1), f,protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(np.array(x_train_2), f,protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(np.array(y_train3),f,protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(words_vec,f,protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
def mkdir_infers(vocab, alist,raw,question):
    x_train_1 = []  # 转变成词语对应的下标
    x_train_2 = []
    items = raw[random.randint(0, len(raw) - 1)]  # 随机选择训练集
    print(len(items[2].split("_")))
    que_re = '_'.join(jieba.cut(question)).split("_")
    rest_que = 52 - len(que_re)
    que_re = '_'.join(que_re).strip() + "<a>_" * rest_que
    for i in range(len(alist)):
        x_train_1.append(encode_sent(vocab, que_re, 100))
        x_train_2.append(encode_sent(vocab, alist[i], 100))
    #     x_train_3.append(encode_sent(vocab, nega, 100))
    return np.array(x_train_1), np.array(x_train_2)
def load_data_val_6(testList, vocab, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        x_train_1.append(encode_sent(vocab, items[2], 100))
        break
        x_train_2.append(encode_sent(vocab, items[3], 100))
        x_train_3.append(encode_sent(vocab, items[3], 100))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_9(trainList, vectors, size):
    x_train_1 = []
    x_train_2 = []
    y_train = []
    for i in range(0, size):
        pos = trainList[random.randint(0, len(trainList) - 1)]
        posItems = pos.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], posItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, posItems[3], posItems[2], 200))
        y_train.append([1, 0])
        neg = trainList[random.randint(0, len(trainList) - 1)]
        negItems = neg.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], negItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, negItems[3], posItems[2], 200))
        y_train.append([0, 1])
    return np.array(x_train_1), np.array(x_train_2), np.array(y_train)

def load_data_val_9(testList, vectors, index):
    x_train_1 = []
    x_train_2 = []
    items = testList[index].split(' ')
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    return np.array(x_train_1), np.array(x_train_2)

def load_data_10(vectors, qalist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = raw[random.randint(0, len(raw) - 1)]
    nega = rand_qa(qalist)
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, nega, items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_11(vectors, qalist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = raw[random.randint(0, len(raw) - 1)]
    nega = rand_qa(qalist)
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, nega, items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
    

if __name__ == '__main__':
    # vector=load_vectors()
    # for key,value in vector.items():
    #     print(key,len(value))\\


    # vocab, train_y, train_x1, train_x2=build_vocab()
    # print(len(train_x1),len(train_x2),len(train_y))
    # load_data_6(train_x1, train_x2, train_y, len(train_y))
    load_word_vec()
    print("555555555555")

    # test_x1,test_x2=build_vocab_test()
    # test_x1,test_x2,word_vec=load_data_temp_test(test_x1,test_x2,len(test_x2))
    # print(len(test_x1))
