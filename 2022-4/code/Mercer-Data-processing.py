#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




hair_dryer_data=pd.read_excel('hair_dryer.xlsx')
microwave_data=pd.read_excel('microwave.xlsx')
pacifier_data=pd.read_excel('pacifier.xlsx')




#对投票率进行检验
def check_helpful(df):
    check_out=[]
    all_list=[]
    for index,row in df.iterrows():
        x=2*row['helpful_votes']-row['total_votes']
        all_list.append(1/(1+np.exp(-1*x)))
        if row['total_votes']!=0:
            if 1/(1+np.exp(-1*x))<0.5:
                check_out.append(index)
    print(check_out)
    print(all_list)
    return check_out,all_list




a,b=check_helpful(hair_dryer_data)
e,f=check_helpful(microwave_data)
c,d=check_helpful(pacifier_data)




helpful_1=pd.DataFrame(b)
helpful_2=pd.DataFrame(d)
helpful_3=pd.DataFrame(f)




a=pd.DataFrame(b)
c=pd.DataFrame(d)
e=pd.DataFrame(e)
helpful_rate=pd.concat([a,c],axis=1)
helpful_rate=pd.concat([helpful_rate,e],axis=1)
helpful_rate




helpful_rate.columns=['h','p','m']
helpful_rate.to_excel('helpful.xlsx')




hair_dryer_data=pd.read_excel('hair_dryer.xlsx')
microwave_data=pd.read_excel('microwave.xlsx')
pacifier_data=pd.read_excel('pacifier.xlsx')




#对各个商品的helpful进行可视化
plt.figure(dpi=120)
plt.hist(helpful_1,color="b",alpha=0.5, label='hair_dryer_helpful')
plt.hist(helpful_2,color="r",alpha=0.5, label='pacifier_data_helpful')
plt.hist(helpful_3,color="g",alpha=0.5, label='microwave_helpful')
plt.legend(loc='upper left')
plt.show()




import seaborn as sns
sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
plt.figure(dpi=120)
sns.distplot(helpful_,hist=True,bins=10,color="#098154")




print(len(a))
hair_dryer_data=hair_dryer_data.drop(a,axis=0)
data2=pacifier_data.drop(c,axis=0)
data3=microwave_data.drop(e,axis=0)
data2.reset_index(inplace=True,drop=True)
data3.reset_index(inplace=True,drop=True)
data2.to_excel('pacifier_data_1.xlsx')
data3.to_excel('microwave_data_1.xlsx')




hair_dryer_data=hair_dryer_data.drop(a,axis=0)




data2=pacifier_data.drop(c,axis=0)
data3=microwave_data.drop(e,axis=0)
data2.reset_index(inplace=True,drop=True)
data3.reset_index(inplace=True,drop=True)
hair_dryer_data.reset_index(inplace=True,drop=True)
data2.to_excel('pacifier_data_1.xlsx')
data3.to_excel('microwave_data_1.xlsx')
hair_dryer_data.to_excel('hair_dryer_data_1.xlsx')




#对评论时间进行转化  
def time_count(df):
    time_={}
    time_got=[str(i) for i in range(1,12)]
    for row in df['review_date']:
        time_list=row.split('/')
        time=time_list[0]
        if time in time_.keys():
            time_[time]+=1
        else:
            time_[time]=1
    return time_




time_count1=time_count(hair_dryer_data)
time_count2=time_count(pacifier_data)
time_count3=time_count(microwave_data)




time_count1=sorted(time_count1.items(), key=lambda time_count1:int(time_count1[0]),reverse = False)
time_count2=sorted(time_count2.items(), key=lambda time_count2:int(time_count2[0]),reverse = False)
time_count3=sorted(time_count3.items(), key=lambda time_count3:int(time_count3[0]),reverse = False)




def get_(lis):
    target=[]
    for i in lis:
        target.append(i[-1])
    return target




time_count1=get_(time_count1)
time_count2=get_(time_count2)
time_count3=get_(time_count3)




time_count1=pd.Series(time_count1)
time_count2=pd.Series(time_count2)
time_count3=pd.Series(time_count3)




all_=pd.concat([time_count1,time_count2],axis=1)
all_=pd.concat([all_,time_count3],axis=1)
print(all_)




#对每个月份的评论数进行可视化
sns.lineplot(x=time_count1.index+1,y=time_count1)
sns.lineplot(x=time_count2.index+1,y=time_count2)
sns.lineplot(x=time_count3.index+1,y=time_count3)





def time_trans(df):
    time_=[]
    for row in df['review_date']:
        time_list=row.split('/')
        time=int(time_list[2]+time_list[0].zfill(2)+time_list[1].zfill(2))
        time_.append(time)
    return time_




time_=time_trans(hair_dryer_data)
print(time_)
time_=pd.DataFrame(time_)
print(time_)




hair_dryer_data['review_date'].head(10)




from scipy.linalg import norm
def tfidf_similarity(s1, s2):

    def add_space(s):

        return ' '.join(list(s))

    

    # 将字中间加入空格

    s1, s2 = add_space(s1), add_space(s2)

    # 转化为TF矩阵

    cv = TfidfVectorizer(tokenizer=lambda s: s.split())

    corpus = [s1, s2]

    vectors = cv.fit_transform(corpus).toarray()

    # 计算TF系数

    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

 

 

s1 = 'It is great'

s2 = 'It is really bad'

print(tfidf_similarity(s1, s2))




from gensim.models import word2vec





import numpy as np
from collections import defaultdict
 
 
class word2vec():
 
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
 
    def generate_training_data(self, settings, corpus):
        """
        得到训练数据
        """
 
        #defaultdict(int)  一个字典，当所访问的键不存在时，用int类型实例化一个默认值
        word_counts = defaultdict(int)
 
        #遍历语料库corpus
        for row in corpus:
            for word in row:
                #统计每个单词出现的次数
                word_counts[word] += 1
 
        # 词汇表的长度
        self.v_count = len(word_counts.keys())
        # 在词汇表中的单词组成的列表
        self.words_list = list(word_counts.keys())
        # 以词汇表中单词为key，索引为value的字典数据
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        #以索引为key，以词汇表中单词为value的字典数据
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
 
        training_data = []
 
        for sentence in corpus:
            sent_len = len(sentence)
 
            for i, word in enumerate(sentence):
 
                w_target = self.word2onehot(sentence[i])
 
                w_context = []
 
                for j in range(i - self.window, i + self.window):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
 
                training_data.append([w_target, w_context])
 
        return np.array(training_data)
 
    def word2onehot(self, word):
 
        #将词用onehot编码
 
        word_vec = [0 for i in range(0, self.v_count)]
 
        word_index = self.word_index[word]
 
        word_vec[word_index] = 1
 
        return word_vec
 
    def train(self, training_data):
 
 
        #随机化参数w1,w2
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
 
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
 
        for i in range(self.epochs):
 
            self.loss = 0
 
            # w_t 是表示目标词的one-hot向量
            #w_t -> w_target,w_c ->w_context
            for w_t, w_c in training_data:
 
                #前向传播
                y_pred, h, u = self.forward(w_t)
 
                #计算误差
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
 
                #反向传播，更新参数
                self.backprop(EI, h, w_t)
 
                #计算总损失
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
 
            print('Epoch:', i, "Loss:", self.loss)
 
    def forward(self, x):
        """
        前向传播
        """
 
        h = np.dot(self.w1.T, x)
 
        u = np.dot(self.w2.T, h)
 
        y_c = self.softmax(u)
 
        return y_c, h, u
 
 
    def softmax(self, x):
        """
        """
        e_x = np.exp(x - np.max(x))
 
        return e_x / np.sum(e_x)
 
 
    def backprop(self, e, h, x):
 
        d1_dw2 = np.outer(h, e)
        d1_dw1 = np.outer(x, np.dot(self.w2, e.T))
 
        self.w1 = self.w1 - (self.lr * d1_dw1)
        self.w2 = self.w2 - (self.lr * d1_dw2)
 
    def word_vec(self, word):
 
        """
        获取词向量
        通过获取词的索引直接在权重向量中找
        """
 
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
 
        return v_w
 
    def vec_sim(self, word, top_n):
        """
        找相似的词
        """
 
        v_w1 = self.word_vec(word)
        word_sim = {}
 
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
 
            #np.linalg.norm(v_w1) 求范数 默认为2范数，即平方和的二次开方
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
 
            word = self.index_word[i]
            word_sim[word] = theta
 
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
 
        for word, sim in words_sorted[:top_n]:
            print(word, sim)
 
    def get_w(self):
        w1 = self.w1
        return  w1
#超参数
settings = {
    'window_size': 2,   #窗口尺寸 m
    #单词嵌入(word embedding)的维度,维度也是隐藏层的大小。
    'n': 10,
    'epochs': 50,         #表示遍历整个样本的次数。在每个epoch中，我们循环通过一遍训练集的样本。
    'learning_rate':0.01 #学习率
}
 
#数据准备
text = "natural language processing and machine learning is fun and exciting"
#按照单词间的空格对我们的语料库进行分词
corpus = [[word.lower() for word in text.split()]]
print(corpus)
 
#初始化一个word2vec对象
w2v = word2vec()
 
training_data = w2v.generate_training_data(settings,corpus)
 
#训练
w2v.train(training_data)
 
# 获取词的向量
word = "machine"
vec = w2v.word_vec(word)
print(word, vec)
 
# 找相似的词
w2v.vec_sim("machine", 3)





import numpy as np
from collections import defaultdict
 
 
class word2vec():
 
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
 
    def generate_training_data(self, settings, corpus):
        """
        得到训练数据
        """
 
        #defaultdict(int)  一个字典，当所访问的键不存在时，用int类型实例化一个默认值
        word_counts = defaultdict(int)
 
        #遍历语料库corpus
        for row in corpus:
            for word in row:
                #统计每个单词出现的次数
                word_counts[word] += 1
 
        # 词汇表的长度
        self.v_count = len(word_counts.keys())
        # 在词汇表中的单词组成的列表
        self.words_list = list(word_counts.keys())
        # 以词汇表中单词为key，索引为value的字典数据
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        #以索引为key，以词汇表中单词为value的字典数据
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
 
        training_data = []
 
        for sentence in corpus:
            sent_len = len(sentence)
 
            for i, word in enumerate(sentence):
 
                w_target = self.word2onehot(sentence[i])
 
                w_context = []
 
                for j in range(i - self.window, i + self.window):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
 
                training_data.append([w_target, w_context])
 
        return np.array(training_data)
 
    def word2onehot(self, word):
 
        #将词用onehot编码
 
        word_vec = [0 for i in range(0, self.v_count)]
 
        word_index = self.word_index[word]
 
        word_vec[word_index] = 1
 
        return word_vec
 
    def train(self, training_data):
 
 
        #随机化参数w1,w2
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
 
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
 
        for i in range(self.epochs):
 
            self.loss = 0
 
            # w_t 是表示目标词的one-hot向量
            #w_t -> w_target,w_c ->w_context
            for w_t, w_c in training_data:
 
                #前向传播
                y_pred, h, u = self.forward(w_t)
 
                #计算误差
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
 
                #反向传播，更新参数
                self.backprop(EI, h, w_t)
 
                #计算总损失
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
 
            print('Epoch:', i, "Loss:", self.loss)
 
    def forward(self, x):
        """
        前向传播
        """
 
        h = np.dot(self.w1.T, x)
 
        u = np.dot(self.w2.T, h)
 
        y_c = self.softmax(u)
 
        return y_c, h, u
 
 
    def softmax(self, x):
        """
        """
        e_x = np.exp(x - np.max(x))
 
        return e_x / np.sum(e_x)
 
 
    def backprop(self, e, h, x):
 
        d1_dw2 = np.outer(h, e)
        d1_dw1 = np.outer(x, np.dot(self.w2, e.T))
 
        self.w1 = self.w1 - (self.lr * d1_dw1)
        self.w2 = self.w2 - (self.lr * d1_dw2)
 
    def word_vec(self, word):
 
        """
        获取词向量
        通过获取词的索引直接在权重向量中找
        """
 
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
 
        return v_w
 
    def vec_sim(self, word, top_n):
        """
        找相似的词
        """
 
        v_w1 = self.word_vec(word)
        word_sim = {}
 
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
 
            #np.linalg.norm(v_w1) 求范数 默认为2范数，即平方和的二次开方
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
 
            word = self.index_word[i]
            word_sim[word] = theta
 
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
 
        for word, sim in words_sorted[:top_n]:
            print(word, sim)
 
    def get_w(self):
        w1 = self.w1
        return  w1
#超参数
settings = {
    'window_size': 2,   #窗口尺寸 m
    #单词嵌入(word embedding)的维度,维度也是隐藏层的大小。
    'n': 10,
    'epochs': 50,         #表示遍历整个样本的次数。在每个epoch中，我们循环通过一遍训练集的样本。
    'learning_rate':0.01 #学习率
}
 
#数据准备
point_9 = "I love the perfect one,the best great one,pretty good,excellent,awesome,five stars"
point_7 = "I like the nice one,the good one,works well,okay,fine,four stars"
point_5 = "not bad one,just fine,three stars"
point_3="bad one,disappoint,two stars"
point_1="terrible one,garbage,junk,one stars"
#按照单词间的空格对我们的语料库进行分词
corpus = [[word.lower() for word in point_9.split()]]
print(corpus)
 
#初始化一个word2vec对象
w2v = word2vec()
 
training_data = w2v.generate_training_data(settings,corpus)
 
#训练
w2v.train(training_data)
 
# 获取词的向量
word = "great"
#vec = w2v.word_vec(word)
#print(word, vec)
 
# 找相似的词
w2v.vec_sim("five", 5)




point_9 = "I love the perfect one,the best great one,pretty good,excellent,awesome,five stars"
point_7 = "I like the nice one,the good one,works well,okay,fine,four stars"
point_5 = "not bad one,just fine,three stars"
point_3="bad one,disappoint,two stars"
point_1="terrible one,garbage,junk,one stars"




temp=hair_dryer_data[hair_dryer_data['verified_purchase']=='N']
extro=temp[temp['vine']=='N']
print(extro['vine'].value_counts())
print(extro['verified_purchase'].value_counts())




strs='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
list_=[a for a in strs]
i=0
for single in hair_dryer_data['review_headline']:
    i+=1
    if len(set(list_)-set(list(str(single))))==52:
        print(single,i)




strs='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
list_=[a for a in strs]
i=0
for single in microwave_data['review_headline']:
    i+=1
    if len(set(list_)-set(list(str(single))))==52:
        print(single,i)




strs='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
list_=[a for a in strs]
i=0
for single in microwave_data['review_body']:
    i+=1
    if len(set(list_)-set(list(str(single))))==52:
        print(single,i)




strs='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
list_=[a for a in strs]
i=0
for single in pacifier_data['review_headline']:
    i+=1
    if len(set(list_)-set(list(str(single))))==52:
        print(single,i)




import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation




n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(microwave_data['review_headline'])




n_topics = 5
lda = LatentDirichletAllocation( n_components=5,
                                            max_iter=500,
                                learning_method='online',
                                learning_offset=50,random_state=0)
lda.fit(tf)




def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]                        
            for i in topic.argsort()[:-n_top_words - 1:-1]]))




n_top_words = 5
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)




print(lda.score(tf))




import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)






