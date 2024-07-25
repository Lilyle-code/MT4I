import datetime

#starttime = datetime.datetime.now()  # 现在时间

import numpy as np
import cv2
#import cv2.cv as cv
import os
import math


# KNN算法函数
# 参数：待测试图片的特征值、训练集特征值、训练集标签、k
# 返回值：预测类别标签
def kNNClassify(newInput, dataSet, labels, k):
    # step 1：计算距离
    numSamples = dataSet.shape[0]  # shape[0]表示行数，训练集样本数600
    # tile(A, reps): 构造一个矩阵，通过A复制reps次得到
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值600行256列
    #diff = np.tile(newInput, (numSamples, 1)) + dataSet#变体1-1
    squaredDiff = diff ** 2  # 将差值平方
    #squaredDiff = diff * 2#变体2-AOR-2
    #squaredDiff = diff ** 3#变体5-CRP-32
    squaredDist = np.sum(squaredDiff, axis=1)  # 按行累加，转为1行600列
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离
    #distance = squaredDist * 0.5#变体3-AOR-3
    #distance = squaredDist ** 1.5#变体6-CRP-34

    # step 2: 对距离排序
    # argsort() 返回从小到大排序后，各数值排序前的索引值
    sortedDistIndices = np.argsort(distance)  # 1行600列
    classCount = {}  # 计算10个类别出现次数

    for i in range(k):
        # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]  # 找到最近邻的类别标签

        # step 4: 计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        #classCount[voteLabel] = classCount.get(voteLabel, 1) + 1#变体7-CRP-35
        #classCount[voteLabel] = classCount.get(voteLabel, 0) + 2#变体8-CRP-36

    # step 5: 返回出现次数最多的类别标签
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        #if value > maxCount:
        #if not(value > maxCount):#变体4-COI-25
        if value < maxCount:#变体23-51
        #if value >= maxCount:#变体10-ROR-89
            maxCount = value
            maxIndex = key
    return maxIndex


def testAPI(teststr,trainstr):
    # 主程序
    X_test = []  # 测试集特征值600*65536
    Y_test = []  # 测试集标签600*1
    X_train = []  # 训练集特征值400*65536
    Y_train = []  # 训练集标签400*1


    # 遍历10类图片的文件夹
    for i in range(0, 10):
        # 遍历训练集文件夹，读取图片
        for f in os.listdir((trainstr+'/%s') % i):
            # 打开一张图片
            Images = cv2.imread((trainstr+'/%s/%s') % (i, f))
            # 统一为256*256大小
            image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_NEAREST)
            # 传入彩色图像BGR的B通道，0~255的值分为256份计算直方图
            hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
            # 将B通道不同值的数量作为特征值，则有256个特征
            # 将一个图像的256个特征值降到一维，作为一行进行添加
            X_train.append(((hist).flatten()))
            # 添加标签
            Y_train.append(i)

        # 遍历测试集文件夹，读取图片
        for f in os.listdir((teststr+'/%s') % i):
            # 打开一张图片
            Images = cv2.imread((teststr+'/%s/%s') % (i, f))
            # 统一为256*256大小
            image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_NEAREST)
            # 传入彩色图像BGR的B通道，0~255的值分为256份计算直方图
            hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
            # 将B通道不同值的数量作为特征值，则有256个特征
            # 将一个图像的256个特征值降到一维，作为一行进行添加
            X_test.append(((hist).flatten()))
            # 添加标签
            Y_test.append(i)

    # 特征提取结束
    X_train = np.array(X_train)  # 转为数组，600行256列
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)  # 400行256列
    Y_test = np.array(Y_test)

    '''#MR=22:训练集顺序打乱
    if MR == 22:
        state = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(state)
        np.random.shuffle(Y_train)
    #MR=23:特征值做仿射变换
    elif MR == 23:
        num = int(len(X_train[0])/2)
        k = 1/2
        b = 2
        for i in range(0,X_train.shape[0]):
            for j in range(0,num):
                X_train[i][j] = X_train[i][j] * k + b
        for i in range(0,X_test.shape[0]):
            for j in range(0,num):
                X_test[i][j] = X_test[i][j] * k + b
    #MR=24:特征顺序打乱
    elif MR == 24:
        state = np.random.get_state()
        for i in range(0,X_train.shape[0]):
            np.random.shuffle(X_train[i])
            np.random.set_state(state)
        for i in range(0,X_test.shape[0]):
            np.random.shuffle(X_test[i])
            np.random.set_state(state)
    #MR=25:添加无关属性作为特征
    elif MR == 25:
        for i in range(0,X_train.shape[0]):
            X_train[i].append(0)
        for i in range(0,X_test.shape[0]):
            X_test[i].append(0)'''

    # 分类
    predictions = []  # 400个测试图片的预测标签
    # 遍历400个测试图片，shape[0]取行数
    for i in range(X_test.shape[0]):
        # 输入待测试图片的特征值、训练集特征值、训练集标签、k=10
        predictions_labes = kNNClassify(X_test[i], X_train, Y_train, 5)
        predictions.append(predictions_labes)

    '''# 比较预测标签与实际标签
    delta = Y_test - predictions  # 计算预测标签与实际标签差值
    num = Y_test.shape[0]  # 测试集数量
    right = 0  # 预测正确的数量
    right_num = []  # 各类别预测正确的数量，初始为0
    for i in range(0, 10):
        right_num.append(0)

    # 遍历差值，差值为0表示预测正确
    for i in range(Y_test.shape[0]):
        if (delta[i] == 0):
            right = right + 1
            right_num[i % 10] = right_num[i % 10] + 1

    # 计算成功率
    success = right * 1.0 / num
    print(success)
    print("测试集数量：",num)
    print("预测类别正确数量：",right)
    print("每一类别预测正确数量：")
    for i in range(0, 10):
    print(right_num[i])
    print("\n成功率：",success)

    # 计算程序运行时间
    endtime = datetime.datetime.now()
    print("程序运行时间：",endtime - starttime)

    #cv2.waitKey(0)  # 无限期等待键盘输入'''
    return predictions
