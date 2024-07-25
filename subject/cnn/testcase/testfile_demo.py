import numpy as np
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import graph_util
import random

def testthreeAPI(teststr,trainstr,MR):


    path_ = trainstr
    classes = ['0', '1', '2', '3', '4']
    all = []

    for index, name in enumerate(classes):
        path = path_ + name + '/'

        path_all = glob.glob(path + '*.png')

        label = [0, 0, 0, 0, 0]
        label[index] = 1

        for img_path in path_all:
            img = Image.open(img_path)

            img = img.convert('L')

            img = img.resize((28, 28))

            img = img.point(lambda i: 255 - i)

            data_i = [np.array(img).flatten().tolist(), label]
            all.append(data_i)

    random.shuffle(all)
    all = np.array(all)

    img = all[:, 0]
    label = all[:, 1]

    img = img.tolist()
    label = label.tolist()



    learning_rate=0.0005
    batch_size=16
    display_step=1


    n_input=784
    n_classes=5
    dropout=0.8


    x=tf.placeholder(tf.float32,[None,n_input], name='input_x')
    y=tf.placeholder(tf.float32,[None,n_classes], name='input_y')
    keep_prob=tf.placeholder(tf.float32,)


    def conv2d(x,W,b,strides=1):
        x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
        x=tf.nn.bias_add(x,b)
        return tf.nn.relu(x)

    def maxpool2d(x,k=2):
        return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')


    def norm(pool1,lsize=4):
        return tf.nn.lrn(pool1,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75)


    weights={
        'wc1':tf.Variable(tf.random_normal([11,11,1,96],stddev=0.01)),
        'wc2':tf.Variable(tf.random_normal([5,5,96,256],stddev=0.01)),
        'wc3':tf.Variable(tf.random_normal([3,3,256,384],stddev=0.01)),
        'wc4':tf.Variable(tf.random_normal([3,3,384,384],stddev=0.01)),
        'wc5':tf.Variable(tf.random_normal([3,3,384,256],stddev=0.01)),
        'wd1':tf.Variable(tf.random_normal([2*2*256,4096],stddev=0.01)),
        'wd2':tf.Variable(tf.random_normal([4096,4096],stddev=0.01)),
        'out':tf.Variable(tf.random_normal([4096,n_classes],stddev=0.01))
    }
    biases={
        'bc1':tf.Variable(tf.random_normal([96],stddev=0.01)),
        'bc2':tf.Variable(tf.random_normal([256],stddev=0.01)),
        'bc3':tf.Variable(tf.random_normal([384],stddev=0.01)),
        'bc4':tf.Variable(tf.random_normal([384],stddev=0.01)),
        'bc5':tf.Variable(tf.random_normal([256],stddev=0.01)),
        'bd1':tf.Variable(tf.random_normal([4096],stddev=0.01)),
        'bd2':tf.Variable(tf.random_normal([4096],stddev=0.01)),
        'out':tf.Variable(tf.random_normal([n_classes]))
    }



    def alex_net(x, weights, biases, dropout):

        x = tf.reshape(x, shape=[-1, 28, 28, 1])



        conv1 = conv2d(x, weights['wc1'], biases['bc1'])

        pool1 = maxpool2d(conv1, k=2)


        norm1 = norm(pool1)


        conv2 = conv2d(norm1, weights['wc2'], biases['bc2'])

        pool2 = maxpool2d(conv2, k=2)
        norm2 = norm(pool2)


        conv3 = conv2d(norm2, weights['wc3'], biases['bc3'])

        pool3 = maxpool2d(conv3, k=2)
        norm3 = norm(pool3)


        conv4 = conv2d(norm3, weights['wc4'], biases['bc4'])

        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])

        pool5 = maxpool2d(conv5, k=2)
        norm5 = norm(pool5)




        fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        fc1 = tf.nn.dropout(fc1, dropout)



        fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
        fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)

        fc2 = tf.nn.dropout(fc2, dropout)


        return tf.add(tf.matmul(fc2, weights['out']), biases['out'],name='pred')



    pred=alex_net(x,weights,biases,keep_prob)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
    optim=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    saver = tf.train.Saver()

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step*batch_size < len(label):
            batch_x,batch_y = img[(step-1)*batch_size:step*batch_size], label[(step-1)*batch_size:step*batch_size]
            sess.run(optim,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            if step % display_step==0:

                loss, acc_num = sess.run([cost,acc],feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
                print('Iter:%d,Loss:%f,Train Acc:%f'%(step*batch_size,loss,acc_num))
            step+=1


        saver.save(sess, 'checkpoint/model.ckpt')
        print('Optimization finished')
        path_ = teststr

        classes = ['0', '1', '2', '3', '4']
        all = []

        for index, name in enumerate(classes):
            path = path_ + name + '/'

            path_all = glob.glob(path + '*.png')

            label = [0, 0, 0, 0, 0]
            label[index] = 1

            for img_path in path_all:
                img = Image.open(img_path)

                img = img.convert('L')

                img = img.resize((28, 28))

                img = img.point(lambda i: 255 - i)

                data_i = [np.array(img).flatten().tolist(), label]
                all.append(data_i)


        all = np.array(all)

        img = all[:, 0]
        label = all[:, 1]

        img1 = img.tolist()
        label1 = label.tolist()

        #return sess.run(tf.argmax(pred,1),feed_dict={x: img1, y: label1, keep_prob: dropout})
        sess.run(tf.argmax(pred,1),feed_dict={x: img1, y: label1, keep_prob: dropout})
        return 1