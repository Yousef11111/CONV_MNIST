##########################################################
##Import the required libraries and read the MNIST dataset
##########################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
###########################################
## Set the value of the Parameters
###########################################
learning_rate = 0.01
epochs = 20
batch_size = 256
num_batches = int(mnist.train.num_examples/batch_size)
input_height = 28
input_width = 28
n_classes = 10
dropout = 0.75
display_step = 1
filter_height = 5
filter_width = 5
depth_in = 1
depth_out1 = 64
depth_out2 = 128


###########################################
# input output definition
###########################################
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)
###########################################
## Store the weights
## Number of weights of filters to be learnt in 'wc1' => filter_height*filter_width*depth_in*depth_out1
## Number of weights of filters to be learnt in 'wc1' => filter_height*filter_width*depth_out1*depth_out2
## No of Connections to the fully Connected layer => Each maxpooling operation reduces the image size to 1/4.
## So two maxpooling reduces the imase size to /16. There are depth_out2 number of images each of size 1/16 ## of the original image size of input_height*input_width. So there is total of
## (1/16)*input_height* input_width* depth_out2 pixel outputs which when connected to the fully connected layer 
# ## with 1024 units would provide (1/16)*input_height* input_width*depth_out2*1024 connections.
###########################################
weights = {
'wc1' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),
'wc2' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),
#'wd1' : tf.Variable(tf.random_normal([(input_height/4)*(input_height/4)* depth_out2,1024])),
'wd1' : tf.Variable(tf.random_normal([(int(input_height/4))*(int(input_height/4)) * depth_out2,1024])),
'out' : tf.Variable(tf.random_normal([1024,n_classes]))
}
#################################################
## In the 1st Convolutional Layer there are 64 feature maps and that corresponds to 64 biases in 'bc1'
## In the 2nd Convolutional Layer there are 128 feature maps and that corresponds to 128 biases in 'bc2'
## In the Fully Connected Layer there are 1024 units and that corresponds to 1024 biases in 'bd1'
## In the output layet there are 10 classes for the Softmax and that corresponds to 10 biases in 'out'
#################################################
biases = {
'bc1' : tf.Variable(tf.random_normal([64])),
'bc2' : tf.Variable(tf.random_normal([128])),
'bd1' : tf.Variable(tf.random_normal([1024])),
'out' : tf.Variable(tf.random_normal([n_classes]))
}


##################################################
## Create the different layers
##################################################
'''C O N V O L U T I O N L A Y E R'''
def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
''' P O O L I N G L A Y E R'''
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
##################################################
## Create the feed forward model
##################################################
def conv_net(x,weights,biases,dropout):
##################################################
## Reshape the input in the 4 dimensional image
## 1st dimension - image index
## 2nd dimension - height
## 3rd dimension - width
## 4th dimension - depth
    x = tf.reshape(x,shape=[-1,28,28,1])
##################################################
## Convolutional layer 1
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,2)
## Convolutional layer 2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,2)
## Now comes the fully connected layer
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
## Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)
## Output class prediction
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out


#######################################################
# Defining the tensorflow Ops for different activities
#######################################################
pred = conv_net(x,weights,biases,keep_prob)
# Define loss function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
## initializing all variables
init = tf.global_variables_initializer()
####################################################
## Launch the execution Graph
####################################################
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob: 1.})
            if epochs % display_step == 0:
                print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(loss),"Training accuracy","{:.5f}".format(acc))
    print('Optimization Completed')
    y1 = sess.run(pred,feed_dict={x:mnist.test.images[:256],keep_prob: 1})
    test_classes = np.argmax(y1,1)
    print('Testing Accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob: 1}))
    f, a = plt.subplots(1, 10, figsize=(10, 2))
    plt.show()
    for i in range(10):
        a[i].imshow(np.reshape(mnist.test.images[i],(28, 28)))
        plt.show()
        print (test_classes[i])
end_time = time.time()
print('Total processing time:',end_time - start_time)


