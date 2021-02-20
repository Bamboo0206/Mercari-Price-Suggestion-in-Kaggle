#!/usr/bin/env python
# coding: utf-8

# # 直接读取稀疏矩阵
#  ***

# In[1]:


import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack, load_npz


# # 读稀疏矩阵

# In[2]:


X_train_sparse=load_npz('X_train_sparse_cjData.npz')
X_test=load_npz('X_test_cjData.npz')
X_test_submit=load_npz('X_test_submit_cjData.npz')
y_train=pd.read_csv('y_train_cjData.csv', sep=",")
true_price=pd.read_csv('Y_test.csv', sep=",")
X_train_sparse.shape, y_train.shape, X_test.shape, true_price.shape, X_test_submit.shape


# In[3]:


y_train = np.log1p(y_train) #训练价格取对数 （这条代码不能运行多次！！！）


# In[ ]:





# In[ ]:





# # Ridge

# In[4]:


from sklearn.linear_model import Ridge


# In[5]:


def ridgeClassify(train_data, train_label):
    ridgeClf = Ridge(
        solver='auto',
        fit_intercept=True,
        alpha=0.5,
        max_iter=500,
        normalize=False,
        tol=0.05)
    # 训练
    ridgeClf.fit(train_data, train_label)
    return ridgeClf


# # 交叉验证

# In[6]:


ridgeClf = ridgeClassify(X_train_sparse, y_train)


# In[7]:


test_price = ridgeClf.predict(X_test)


# In[8]:


y_pre_true = np.expm1(test_price)


# * 0.241 df['brand_name']+' '+df['category_name']
# * 0.24731811722829436 df['item_description']+' '+df['category_name']
# 
# * 0.23686654378532468 df['item_description']+' '+df['name']     df['brand_name']+' '+df['category_name']
# 
# * 0.24108560740325458 df['item_description']+' '+df['name']     df['item_description']+' '+df['category_name']    df['item_description']+' '+df['item_condition_id']
# 
# * 0.23634671796262421 df['item_description']+' '+df['name']     df['brand_name']+' '+df['category_name'] tv brand_new
# 
# * 0.23277704301764082 品牌+种类  名称+种类   描述+名字  tv  0.23173446002772038
# 
# * 0.23501340567423148 df['name']+' '+df['category_name']+' '+df['item_description']
# 
# * 0.22947310231523774 名称+种类+牌子  描述+名字

# In[9]:


from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(true_price, y_pre_true)
# 0.22869706937857867


# In[10]:


# print(y_pre_true)


# In[11]:


# print(true_price)


# # Keras MLP

# In[ ]:





# In[12]:


# 缩放 标准化y值
from sklearn.preprocessing import StandardScaler
y_scaler=StandardScaler()
y_train=y_scaler.fit_transform(y_train.values.reshape(-1,1))
y_train


# In[13]:


# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


# In[14]:


# keras mlp
# 需要调整的参数：隐层数、节点数、激活函数、是否使用standard scaler、batch_size、迭代次数
model = Sequential([
    Dense(256, input_shape=(X_train_sparse.shape[1],),activation='relu'),
    Dense(64 ,activation='relu'),
    Dense(1)
])
model.compile(loss='mean_squared_error',optimizer=Adam(lr=3e-3))

for i in range(3):
    model.fit(x=X_train_sparse, y=y_train,batch_size=2**(10+i),epochs=1,verbose=1)
    model.save( str(i)+'-cjData_keras_mlp.h5')
    
    #predict
    y_pre_mlp=model.predict(X_test,verbose=1)
    y_pre_mlp=y_scaler.inverse_transform(y_pre_mlp)#缩放
    y_pre_true_mlp = np.expm1(y_pre_mlp)# 还原为正常价格
    MSLE=mean_squared_log_error(true_price, y_pre_true_mlp)
    print(MSLE)
    #结果保存到文件
    file = open(str(i)+'-mlp_testSet.csv',mode = 'w')
    for j in range(len(y_pre_true_mlp)):
        s = str(y_pre_true_mlp[j][0])
        s = s+'\n'
        file.write(s)
    file.write('MSLE='+str(MSLE))
    file.close()
    
    #predict submit
    y_submit_pre_mlp=model.predict(X_test_submit,verbose=1)
    y_submit_pre_mlp=y_scaler.inverse_transform(y_submit_pre_mlp)#缩放
    y_submit_pre_true_mlp = np.expm1(y_submit_pre_mlp)# 还原为正常价格
    # 结果保存到文件
    file = open(str(i)+'-cjData_y_submit_true.csv',mode = 'w')
    for j in range(len(y_submit_pre_true_mlp)):
        s = str(y_submit_pre_true_mlp[j][0])
        s = str(j)+'\t'+s+'\n'
        file.write(s)
    file.close()

