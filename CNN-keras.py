import glob
import time
import csv 
import numpy as np
import pandas as pd
# 载入输入数据
# 从对应文件夹中提取 CSV 列表
# 文件夹中共315个CSV文件
csvx_list = glob.glob('E:/PHM 2010/c1/c1/*.csv')
# 创建储存数据的数组
A = []
# 建立循环 从csvx_list 中读取315个 CSV列表
for i in range( len(csvx_list)):
    # print(csvx_list[1])
    # filename='csvx_list[i]'
    # 第i个csv列表的数据 暂时储存在filename中
    filename = csvx_list[i]
    # 循环 第i个 csv列表中数据
    # 目的是从 csv列表中 每个信号提取5000个数据点， 共35000个数据点
    # 315个列表每个信号的数据点从10w至30w不等
    with open(filename,'r') as f:
        row  = csv.reader(f)
        
        a = []
        # 通过循环，间隔20个数据点取一个点，防止某个 csv列表中的数据过少而造成数据提取失败

        j = 0
        for r in row :
            j += 1

            if j > 500 * 200:
                break
            if not j % 200:
                a.append([float(i) for i in r])
                # np.array(A)

    A.append(a)
# 转换数据类型至 numpy
A = np.array(A)
#A = A.reshape(A.shape[0], A.shape[-1], -1)
X=A
pass

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.utils import np_utils,plot_model

from sklearn.model_selection import cross_val_score,train_test_split

from keras.layers import Dense, Dropout,Flatten,Conv1D,MaxPooling1D

from keras.models import model_from_json

import matplotlib.pyplot as plt

from keras import backend as K



# 载入输出数据

df = pd.read_csv(r"E:\PHM 2010\c1\c1_wear.csv")
Y = df.values[:, 1]
# X = np.expand_dims(A, axis=2)#增加一维轴




 

# 划分训练集，测试集

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

 

# 自定义度量函数

def coeff_determination(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ))

    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

 

# 定义一个神经网络

model = Sequential()

model.add(Conv1D(16, 3,input_shape=(500,7), activation='relu'))

model.add(Conv1D(16, 3, activation='relu'))

model.add(MaxPooling1D(pool_size=3,strides=3))

model.add(Conv1D(64, 3, activation='relu'))

model.add(Conv1D(64, 3, activation='relu'))

model.add(MaxPooling1D(pool_size=3,strides=3))

model.add(Conv1D(128, 3, activation='relu'))

model.add(Conv1D(128, 3, activation='relu'))

model.add(MaxPooling1D(pool_size=3,strides=3))

model.add(Conv1D(64, 3, activation='relu'))

model.add(Conv1D(64, 3, activation='relu'))

model.add(MaxPooling1D(pool_size=3,strides=3))

model.add(Flatten())

# 为网络增加 全连接层 
model.add(Dense(1, activation='linear'))

plot_model(model, to_file='./model_linear.png', show_shapes=True)#

# 打印model，输出网络结构
print(model.summary())

# 定义model的优化器类型，损失函数类型，调用度量函数
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])


# 训练模型
# 前两项为训练集，validation 为验证集， 迭代次数40代，批大小为10
model.fit(X_train,Y_train, validation_data=(X_test, Y_test),epochs=40, batch_size=10,verbose=2)

 
# 准确率

scores = model.evaluate(X_test,Y_test,verbose=1)

print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

 

# 预测值散点图

predicted = model.predict(X_test)
print('print the result of predicted:',predicted)
plt.scatter(Y_test,predicted)
 # np.linspace(x1,x2,t) 从 (x1,x2)中等差数列提取t个点
x=np.linspace(0,170,18)
 # 这里改过了，图像可以显示了
y=x

plt.plot(x,y,color='red',linewidth=1.0,linestyle='--',label='line')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.legend(["y = x","磨损预测值"])

plt.title("预测值与真实值的偏离程度")

plt.xlabel('真实磨损值')

plt.ylabel('磨损预测值')
 # 保存图像
plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)

plt.show()

plt.figure(2)
x_label = range(1,len(Y_test)+1)
plt.plot(x_label,Y_test,color='blue')
plt.plot(x_label,predicted,color='green')

 

# 计算误差

result =abs(np.mean(predicted - Y_test))

print("The mean error of linear regression:")

print(result)



            
