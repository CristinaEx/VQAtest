from data_reader import DataReader

# VGG19网络模型
class VGG19model:
    def __init__(self):
        pass

    def getRandomWeightsAndBiases(self, shape, predict_num = 10, div = 100):
        """
        shape = [w,h,mod]
        w,h需为32倍数
        predict_num:预测的类别数量
        div:权值和偏值 /= div(防止梯度爆炸)
        获取随机的VGG16的权重和偏值
        """
        weights = {  
            # 3x3 conv, 3 input, 24 outputs  
            'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),  
            'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64])),  
            # pool
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),  
            'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128])),  
            # pool  
            'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256])),  
            'wc6': tf.Variable(tf.random_normal([3, 3, 256, 256])),  
            'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256])),  
            'wc8': tf.Variable(tf.random_normal([3, 3, 256, 256])),  
            # pool
            'wc9': tf.Variable(tf.random_normal([3, 3, 256, 512])),  
            'wc10': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc11': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc12': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            # pool
            'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc14': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc15': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc16': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            # pool
            # fully connected, 32*32*96 inputs, 4096 outputs  
            'wd1': tf.Variable(tf.random_normal([int(shape[0] * shape[1] / 2), 4096])),  
            'wd2': tf.Variable(tf.random_normal([4096, 4096])),  
            # 1024 inputs, predict_num outputs (class prediction)  
            'out': tf.Variable(tf.random_normal([4096, predict_num]))
        }
        biases = {  
            'bc1': tf.Variable(tf.random_normal([64])),  
            'bc2': tf.Variable(tf.random_normal([64])),  
            # pool
            'bc3': tf.Variable(tf.random_normal([128])),  
            'bc4': tf.Variable(tf.random_normal([128])),
            # pool
            'bc5': tf.Variable(tf.random_normal([256])),  
            'bc6': tf.Variable(tf.random_normal([256])),  
            'bc7': tf.Variable(tf.random_normal([256])), 
            'bc8': tf.Variable(tf.random_normal([256])),  
            # pool
            'bc9': tf.Variable(tf.random_normal([512])),  
            'bc10': tf.Variable(tf.random_normal([512])),
            'bc11': tf.Variable(tf.random_normal([512])),  
            'bc12': tf.Variable(tf.random_normal([512])),  
            # pool
            'bc13': tf.Variable(tf.random_normal([512])), 
            'bc14': tf.Variable(tf.random_normal([512])),
            'bc15': tf.Variable(tf.random_normal([512])),
            'bc16': tf.Variable(tf.random_normal([512])),
            # fully connected, 32*32*96 inputs, 4096 outputs  
            'bd1': tf.Variable(tf.random_normal([4096])),  
            'bd2': tf.Variable(tf.random_normal([4096])), 
            # 4096 inputs, predict_num outputs (class prediction)  
            'out': tf.Variable(tf.random_normal([predict_num]))  
        }
        # 防止梯度爆炸
        for key in weights.keys():
            weights[key] = weights[key] / div
        for key in biases.keys():
            biases[key] = biases[key] / div
        return weights,biases

    def conv2d(self, data, weights, biase, strides=1): 
        """
        卷积函数设置:
        data:数据
        weights:权重
        biase:偏差值
        strides:步长
        返回:
        卷积->RELU->返回值
        """ 
        # Conv2D wrapper, with bias and relu activation  
        data = tf.nn.conv2d(data, weights, strides=[1, strides, strides, 1], padding='SAME')  
        data = tf.nn.bias_add(data, biase)  
        return tf.nn.relu(data)  

    def maxpool2d(self, data, k=2):  
        """
        池化:
        data:数据
        k:池化层大小(1,k,k,1)
        步长:(1,k,k,1)
        """
        # MaxPool2D wrapper  
        return tf.nn.max_pool(data, ksize=[1, k, k, 1], strides=[1, k, k, 1],  
                          padding='SAME')  

    def getNet(self, data, weights, biases, dropout, output_hiding = false):  
        """
        data:数据(32N+,32N+,3)
        weights:权重
        biases:偏置
        dropout:防止过度拟合(float32)
        output_hiding:是否输出隐藏层(维度为[pic_num,4096])
        return out(输出层[pic_num,predict_num],数据含义是给每张图可能对应于的每个模型进行打分)
        """
        # Convolution Layer  
        conv1 = self.conv2d(data, weights['wc1'], biases['bc1'])  
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])  
        # Max Pooling (down-sampling)  
        pool1 = self.maxpool2d(conv2, k=2)  
        # Convolution Layer  
        conv3 = self.conv2d(pool1, weights['wc3'], biases['bc3'])  
        conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'])  
        # Max Pooling (down-sampling)  
        pool2 = self.maxpool2d(conv4, k=2)  
        # Convolution Layer  
        conv5 = self.conv2d(pool2, weights['wc5'], biases['bc5'])  
        conv6 = self.conv2d(conv5, weights['wc6'], biases['bc6'])  
        conv7 = self.conv2d(conv6, weights['wc7'], biases['bc7']) 
        conv8 = self.conv2d(conv7, weights['wc8'], biases['bc8'])  
        # Max Pooling  
        pool3 = self.maxpool2d(conv8, k=2)  
        # Convolution Layer   
        conv9 = self.conv2d(pool3, weights['wc9'], biases['bc9'])  
        conv10 = self.conv2d(conv9, weights['wc10'], biases['bc10'])   
        conv11 = self.conv2d(conv10, weights['wc11'], biases['bc11'])  
        conv12 = self.conv2d(conv11, weights['wc12'], biases['bc12'])  
        # Max Pooling  
        pool4 = self.maxpool2d(conv12, k=2) 
        conv13 = self.conv2d(pool4, weights['wc13'], biases['bc13'])  
        conv14 = self.conv2d(conv13, weights['wc14'], biases['bc14']) 
        conv15 = self.conv2d(conv14, weights['wc15'], biases['bc15']) 
        conv16 = self.conv2d(conv15, weights['wc16'], biases['bc16']) 
        # Max Pooling  
        pool5 = self.maxpool2d(conv16, k=2)  
        # Fully connected layer  
        # Reshape conv2 output to fit fully connected layer input  
        fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])  
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])  
        fc1 = tf.nn.relu(fc1)  
        # Apply Dropout  
        fc1 = tf.nn.dropout(fc1, dropout)  
        #fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])  
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])  
        fc2 = tf.nn.relu(fc2)  
        # Apply Dropout  
        fc2 = tf.nn.dropout(fc2, dropout)  
        if output_hiding:
            return fc2
        ''''' 
        fc3 = tf.reshape(fc2, [-1, weights['out'].get_shape().as_list()[0]]) 
        fc3 = tf.add(tf.matmul(fc2, weights['out']), biases['bd2']) 
        fc3 = tf.nn.relu(fc2) 
        '''  
        # Output, class prediction  
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])  
        return out

class TrainNetForVQA:
    
    def __init__(self):
        pass

if __name__ == '__main__':
    pass