import numpy as np
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import graph_util
#import random

def testAPI(teststr,trainstr):
    tf.keras.backend.clear_session()

    #设置随机数
    tf.set_random_seed(1)
    np.random.seed(1)

    #path_ = 'train'
    path_ = trainstr
    classes = ['0', '1', '2', '3', '4']
    #classes = ['mutpy', '1', '2', '3', '4']#变体8-CRP-48
    #classes = ['', '1', '2', '3', '4']#变体9-CRP-49
    #classes = ['0', 'mutpy', '2', '3', '4']#变体10-CRP-50
    #classes = ['0', '', '2', '3', '4']#变体11-CRP-51
    #classes = ['0', '1', 'mutpy', '3', '4']#变体12-CRP-52
    #classes = ['0', '1', '', '3', '4']#变体13-CRP-53
    #classes = ['0', '1', '2', 'mutpy', '4']#变体14-CRP-54
    #classes = ['0', '1', '2', '', '4']#变体15-CRP-55
    #classes = ['0', '1', '2', '3', 'mutpy']#变体16-CRP-56
    #classes = ['0', '1', '2', '3', '']#变体17-CRP-57
    all = []
    # 遍历主文件夹下所有的类别文件夹
    for index, name in enumerate(classes):
        path = path_ + '/' + name + '/'
        # 获取所有该类别文件夹下所有的图片路径
        path_all = glob.glob(path + '*.png')
        # 生成label标签
        label = [0, 0, 0, 0, 0]
        #label = [1, 0, 0, 0, 0]#变体18-CRP-62
        #label = [0, 1, 0, 0, 0]#变体19-CRP-63
        #label = [0, 0, 1, 0, 0]#变体20-CRP-64
        #label = [0, 0, 0, 1, 0]#变体21-CRP-65
        #label = [0, 0, 0, 0, 1]#变体22-CRP-66
        label[index] = 1
        #label[index] = 2#变体23-CRP-67
        # 读取该文件夹下所有的图片并添加进列表中
        for img_path in path_all:
            img = Image.open(img_path)
            # RGB三色图转化为灰度图
            img = img.convert('L')
            # 尺寸不同的图片全部改为28*28大小的
            img = img.resize((28, 28))
            # 0-255的色值取反
            img = img.point(lambda i: 255 - i)
            #img = img.point(lambda i: 255 + i)#变体1-AOR-7
            #img = img.point(lambda i: 256 - i)#变体24-CRP-72
            # 将28*28的图像转化为784的一维列表，并合并其标签
            data_i = [np.array(img).flatten().tolist(), label]
            all.append(data_i)
    # 打乱数据集
    #seed = 50
    #np.random.seed(seed)
    np.random.shuffle(all)

    '''#MR=22训练集顺序打乱
    if MR == 22:
        np.random.shuffle(all)'''

    #设置随机数
    tf.set_random_seed(1)
    np.random.seed(1)

    all = np.array(all)
    # 分别取出图片数据和label数据
    img = all[:, 0]
    label = all[:, 1]
    # 最终获得2982*784的图片list数据集，和2982*10的标签的list数据集
    img = img.tolist()
    label = label.tolist()


    # 定义网络的超参数
    learning_rate=0.0005 #学习率
    batch_size=16 #每次训练多少数据
    display_step=1  #每多少次显示一下当前状态

    # 定义网络的结构参数
    n_input=784
    n_classes=5 #标签数量
    #dropout=0.8
    dropout=2.0#变体27-90

    #设定数据占位符
    x=tf.placeholder(tf.float32,[None,n_input], name='input_x')
    #x=tf.placeholder(tf.float32,[None,n_input], name='mutpy')#变体25-CRP-81
    y=tf.placeholder(tf.float32,[None,n_classes], name='input_y')
    #y=tf.placeholder(tf.float32,[None,n_classes], name='mutpy')#变体26-CRP-83
    keep_prob=tf.placeholder(tf.float32, name='keep_prob')

    # 定义卷积操作（Conv layer）
    def conv2d(x,W,b,strides=1):
        x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
        x=tf.nn.bias_add(x,b)
        return tf.nn.relu(x)
    # 定义池化操作
    def maxpool2d(x,k=2):
    #def maxpool2d(x,k=3):#变体27-CRP-90
        return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

    #局部归一化
    def norm(pool1,lsize=4):
    #def norm(pool1,lsize=5):#变体28-CRP-97
        return tf.nn.lrn(pool1,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75)
        #return tf.nn.lrn(pool1,lsize,bias=1.0,alpha=0.001//9.0,beta=0.75)#变体2-AOR-8
        #return tf.nn.lrn(pool1,lsize,bias=1.0,alpha=0.001 * 9.0,beta=0.75)#变体3-AOR-9
        #return tf.nn.lrn(pool1,lsize,bias=2.0,alpha=0.001/9.0,beta=0.75)#变体29-CRP-98
        #return tf.nn.lrn(pool1, lsize, bias=1.0, alpha=1.001 / 9.0, beta=0.75)#变体30-CRP-99
        #return tf.nn.lrn(pool1, lsize, bias=1.0, alpha=0.001 / 10.0, beta=0.75)#变体31-CRP-100
        #return tf.nn.lrn(pool1, lsize, bias=1.0, alpha=0.001 / 9.0, beta=1.75)#变体32-CRP-101

    # 定义网络的权重和偏置参数
    weights={
        'wc1':tf.Variable(tf.random_normal([11,11,1,96],stddev=0.01,seed=1)),
        #'wc1': tf.Variable(tf.random_normal([12, 11, 1, 96], stddev=0.01, seed=1)),#变体33-CRP-118
        #'wc1': tf.Variable(tf.random_normal([11, 12, 1, 96], stddev=0.01, seed=1)),#变体34-CRP-119
        #'wc1': tf.Variable(tf.random_normal([11, 11, 2, 96], stddev=0.01, seed=1)),#变体35-CRP-120
        #'wc1': tf.Variable(tf.random_normal([11, 11, 1, 97], stddev=0.01, seed=1)),#变体36-CRP-121
        #'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96], stddev=1.01, seed=1)),#变体37-CRP-122
        'wc2':tf.Variable(tf.random_normal([5,5,96,256],stddev=0.01,seed=1)),
        #'wc2':tf.Variable(tf.random_normal([6,5,96,256],stddev=0.01,seed=1)),#变体38-CRP-123
        #'wc2':tf.Variable(tf.random_normal([5,6,96,256],stddev=0.01,seed=1)),#变体39-CRP-124
        #'wc2':tf.Variable(tf.random_normal([5,5,96,256],stddev=1.01,seed=1)),#变体40-CRP-127
        'wc3':tf.Variable(tf.random_normal([3,3,256,384],stddev=0.01,seed=1)),
        #'wc3':tf.Variable(tf.random_normal(4,3,256,384],stddev=0.01,seed=1)),#变体41-CRP-128
        #'wc3':tf.Variable(tf.random_normal([3,4,256,384],stddev=0.01,seed=1)),#变体42-CRP-129
        #'wc3':tf.Variable(tf.random_normal([3,3,256,384],stddev=1.01,seed=1)),#变体43-CRP-132
        'wc4':tf.Variable(tf.random_normal([3,3,384,384],stddev=0.01,seed=1)),
        #'wc4':tf.Variable(tf.random_normal([4,3,384,384],stddev=0.01,seed=1)),#变体44-CRP-133
        #'wc4':tf.Variable(tf.random_normal([3,4,384,384],stddev=0.01,seed=1)),#变体45-CRP-134
        #'wc4':tf.Variable(tf.random_normal([3,3,384,384],stddev=1.01,seed=1)),#变体46-CRP-137
        'wc5':tf.Variable(tf.random_normal([3,3,384,256],stddev=0.01,seed=1)),
        #'wc5':tf.Variable(tf.random_normal([4,3,384,256],stddev=0.01,seed=1)),#变体47-CRP-138
        #'wc5':tf.Variable(tf.random_normal([3,4,384,256],stddev=0.01,seed=1)),#变体48-CRP-139
        #'wc5':tf.Variable(tf.random_normal([3,3,384,256],stddev=1.01,seed=1)),#变体49-CRP-142
        'wd1':tf.Variable(tf.random_normal([2*2*256,4096],stddev=0.01,seed=1)),
        #'wd1': tf.Variable(tf.random_normal([2 ** 2 * 256, 4096], stddev=0.01, seed=1)),#变体4-AOR-12
        #'wd1':tf.Variable(tf.random_normal([2*2*256,4096],stddev=1.01,seed=1)),#变体50-CRP-147
        'wd2':tf.Variable(tf.random_normal([4096,4096],stddev=0.01,seed=1)),
        #'wd2':tf.Variable(tf.random_normal([4096,4096],stddev=1.01,seed=1)),#变体51-CRP-150
        'out':tf.Variable(tf.random_normal([4096,n_classes],stddev=0.01,seed=1))
        #'out':tf.Variable(tf.random_normal([4096,n_classes],stddev=0.01,seed=1))#变体52-CRP-152
    }
    biases={
        'bc1':tf.Variable(tf.random_normal([96],stddev=0.01,seed=1)),
        #'bc1':tf.Variable(tf.random_normal([96],stddev=1.01,seed=1)),#变体53-CRP-170
        'bc2':tf.Variable(tf.random_normal([256],stddev=0.01,seed=1)),
        #'bc2':tf.Variable(tf.random_normal([256],stddev=1.01,seed=1)),#变体54-CRP-172
        'bc3':tf.Variable(tf.random_normal([384],stddev=0.01,seed=1)),
        #'bc3':tf.Variable(tf.random_normal([384],stddev=1.01,seed=1)),#变体55-CRP-174
        'bc4':tf.Variable(tf.random_normal([384],stddev=0.01,seed=1)),
        #'bc4':tf.Variable(tf.random_normal([384],stddev=1.01,seed=1)),#变体56-CRP-176
        'bc5':tf.Variable(tf.random_normal([256],stddev=0.01,seed=1)),
        #'bc5':tf.Variable(tf.random_normal([256],stddev=1.01,seed=1)),#变体57-CRP-178
        'bd1':tf.Variable(tf.random_normal([4096],stddev=0.01,seed=1)),
        #'bd1':tf.Variable(tf.random_normal([4096],stddev=1.01,seed=1)),#变体58-CRP-180
        'bd2':tf.Variable(tf.random_normal([4096],stddev=0.01,seed=1)),
        #'bd2':tf.Variable(tf.random_normal([4096],stddev=1.01,seed=1)),#变体59-CRP-182
        'out':tf.Variable(tf.random_normal([n_classes],seed=1))
    }


    # 定义Alexnet网络结构
    def alex_net(x, weights, biases, dropout):
        # 输出的数据做reshape
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # 第一层卷积计算（conv+relu+pool）
        # 卷积
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # 池化
        pool1 = maxpool2d(conv1, k=2)
        #pool1 = maxpool2d(conv1, k=3)#变体60-CRP-191
        # 规范化，局部归一化
        # 局部归一化是仿造生物学上的活跃的神经元对相邻神经元的抑制现象
        norm1 = norm(pool1)

        # 第二层卷积
        conv2 = conv2d(norm1, weights['wc2'], biases['bc2'])
        # 池化
        pool2 = maxpool2d(conv2, k=2)
        #pool2 = maxpool2d(conv2, k=3)#变体61-CRP-196
        norm2 = norm(pool2)

        # 第三层卷积
        conv3 = conv2d(norm2, weights['wc3'], biases['bc3'])
        # 池化
        pool3 = maxpool2d(conv3, k=2)
        #pool3 = maxpool2d(conv3, k=3)#变体62-CRP-201
        norm3 = norm(pool3)

        # 第四层卷积
        conv4 = conv2d(norm3, weights['wc4'], biases['bc4'])
        # 第五层卷积
        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
        # 池化
        pool5 = maxpool2d(conv5, k=2)
        #pool5 = maxpool2d(conv5, k=3)#变体63-CRP-210
        norm5 = norm(pool5)
        # 可以再加上dropout

        # 全连接1
        # 向量化
        fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # 全连接2
        # 向量化
        fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
        #fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[1]])#变体64-CRP-222
        fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)
        # dropout
        fc2 = tf.nn.dropout(fc2, dropout)

        # out
        return tf.add(tf.matmul(fc2, weights['out']), biases['out'],name='pred')
        #return tf.add(tf.matmul(fc2, weights['out']), biases['out'],name='mutpy')#变体65-CRP-231

    # 1.定义损失函数和优化器，并构建评估函数
    # （1）构建模型
    pred=alex_net(x,weights,biases,keep_prob)
    # (2)损失函数和优化器
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
    optim=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #(3)评估函数
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    #保存模型
    saver = tf.train.Saver()
    # 训练
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        #step = 2#变体66-CRP-235
        while step*batch_size < len(label):
        #while step ** batch_size < len(label):#变体5-AOR-21
        #while not(step * batch_size < len(label)):#变体7-COI-46
        #while step*batch_size > len(label):#变体85-ROR-275
        #while step*batch_size <= len(label):#变体86-ROR-276
            batch_x,batch_y = img[(step-1)*batch_size:step*batch_size], label[(step-1)*batch_size:step*batch_size]
            sess.run(optim,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            if step % display_step==0:
                # 显示一下当前的损失和正确率
                loss, acc_num = sess.run([cost,acc],feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
                print('Iter:%d,Loss:%f,Train Acc:%f'%(step*batch_size,loss,acc_num))
            step+=1
            #step+=2#变体67-CRP-241


        #saver.save(sess, 'checkpoint/model.ckpt')
        print('Optimization finished')
        path_ = teststr
        #path_ = 'test'
        classes = ['0', '1', '2', '3', '4']
        all = []
        # 遍历主文件夹下所有的类别文件夹
        for index, name in enumerate(classes):
            path = path_ + '/' + name + '/'
            # 获取所有该类别文件夹下所有的图片路径
            path_all = glob.glob(path + '*.png')
            # 生成label标签
            label = [0, 0, 0, 0, 0]
            #label = [1, 0, 0, 0, 0]#变体78-CRP-260
            #label = [0, 1, 0, 0, 0]#变体79-CRP-261
            #label = [0, 0, 1, 0, 0]#变体80-CRP-262
            #label = [0, 0, 0, 1, 0]#变体81-CRP-263
            #label = [0, 0, 0, 0, 1]#变体82-CRP-264
            label[index] = 1
            #label[index] = 2#变体83-CRP-265
            # 读取该文件夹下所有的图片并添加进列表中
            for img_path in path_all:
                img = Image.open(img_path)
                # RGB三色图转化为灰度图
                img = img.convert('L')
                # 尺寸不同的图片全部改为28*28大小的
                img = img.resize((28, 28))
                # 0-255的色值取反
                img = img.point(lambda i: 255 - i)
                #img = img.point(lambda i: 255 + i)#变体6-AOR-44
                #img = img.point(lambda i: 256 - i)#变体84-CRP-270

                # 将28*28的图像转化为784的一维列表，并合并其标签
                data_i = [np.array(img).flatten().tolist(), label]
                all.append(data_i)
        # 打乱数据集
        #random.shuffle(all)
        all = np.array(all)
        # 分别取出图片数据和label数据
        img = all[:, 0]
        label = all[:, 1]
        # 最终获得2982*784的图片list数据集，和2982*10的标签的list数据集
        img1 = img.tolist()
        label1 = label.tolist()
        '''test_acc = sess.run(acc, feed_dict={x: img1, y: label1, keep_prob: dropout})
        print('Test Acc:%f' %  test_acc)'''
        '''b = tf.argmax(label1,1)
        b = b.eval()
        a,test_acc =  sess.run([tf.argmax(pred,1),acc],feed_dict={x: img1, y: label1, keep_prob: dropout})
        c=0
        for i in range(0,300):
            if a[i]!=b[i]:
                c=c+1
        d=c/300'''
        return sess.run(tf.argmax(pred,1),feed_dict={x: img1, y: label1, keep_prob: dropout})
