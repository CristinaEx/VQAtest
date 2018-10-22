from data_reader import DataReader
from data_dealer import DataDealer
from tensorflow.python.ops import rnn_cell_impl
from path_set import *
from vgg19 import VGG19model
import tensorflow as tf
import os
import time
import numpy
import json
import psutil
# 防跳出警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用VIS(VGG19)+LSTM网络
# VGG19输入的图像数据为224*224
class TrainNetForVQA:
    
    def __init__(self):
        # 获取字典
        self.dealer = DataDealer(QUESTIONS_DICT_PATH)
        # 使用 basic LSTM Cell.
        # 输入维度:256
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=0.5, state_is_tuple=True)
        if os.path.exists(RUN_DATA_RECORDER):
            # 导入权值
            data = self.__loadData()
            pos = data['pos']
            self.weights = {
                'connect': tf.Variable(data['weight_connect']),
                'in': tf.Variable(data['weight_in']),
                'out': tf.Variable(data['weight_out'])
            }
            self.biases = {
                'connect': tf.Variable(data['biase_connect']),
                'in': tf.Variable(data['biase_in']),
                'out': tf.Variable(data['biase_out'])
            }
            self.init_state = rnn_cell_impl.LSTMStateTuple(tf.Variable(data['c'],name = 'BasicLSTMCellZeroState_c'),tf.Variable(data['h'],name = 'BasicLSTMCellZeroState_h'))
            # print(init_state)
        else:
            # 生成随机权值
            self.weights = {
                'connect': tf.Variable(tf.random_normal([4096, 256])),
                'in': tf.Variable(tf.random_normal([256,256])),
                'out': tf.Variable(tf.random_normal([256,dealer.getWordNum()]))
            }
            self.biases = {
                'connect': tf.Variable(tf.random_normal([256,])),
                'in': tf.Variable(tf.random_normal([256,])),
                'out': tf.Variable(tf.random_normal([dealer.getWordNum(),]))
            }
            pos = 0
            # batch_size = 1
            # 初始化C，H全为零
            self.init_state = lstm_cell.zero_state(1, dtype=tf.float32)
        # 获取样本集信息
        self.reader = DataReader(TRAIN_DATA_TYPE)
        self.reader.set_pos(pos)

    def __getOnehot(self,pos,value,len_):
        """
        获取onehot标签
        位置为pos[i]的值为value[i]
        其他为0
        pos和value的长度必须相等
        返回列表，长度为len_
        """
        if type(pos) != list:
            pos = [pos]
        if type(value) != list:
            value = [value]
        assert len(pos) == len(value)
        result = []
        index = 0
        for i in range(len_):
            if i in pos:
                result.append(value[index])
                index = index + 1
            else:
                result.append(0)
        return result

    def __getVGG19Result(self,img_batch):
        """
        获取VGG19网络的结果
        img_batch:tf.Variable->[1,224,224,3]
        返回最后一层隐藏层的一维列表
        """
        # 导入VGG19模型
        model = VGG19model()
        weights,biases = model.loadWeightsAndBiases(VGG19_WEIGHTS_AND_BIASE_PATH,False)
        # 这里输出的是最后一个隐藏层
        out = model.getNet(img_batch,weights,biases,0.2,True)
        init = tf.global_variables_initializer() 
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            data = sess.run(out)
            try:
                # 请求线程终止
                coord.request_stop()
            except tf.errors.OutOfRangeError:  
                print ('Done training -- epoch limit reached') 
            finally:
                # 一维列表
                data = data[0]
                # print(data)
                # 请求线程终止
                coord.request_stop()
                coord.join(threads)
        return data

    def __saveData(self,data,pos):
        """
        保存数据
        data为保存了数据的字典
        data['^st'] -> numpy.ndarry
        data['st'] -> tensorflow.python.ops.rnn_cell_impl.LSTMStateTuple
        now_index:为当前图片pos
        """
        # 创建目录
        if not os.path.exists(RUN_DATA_PATH):
            os.mkdir(RUN_DATA_PATH) 
        # 写入文件
        file_name = time.strftime("%y %m %d %H %I %M", time.localtime()) + '.json'
        with open(RUN_DATA_RECORDER,'w') as f:
            f.write('{\n')
            f.write('   \"newest_data\":\"' + file_name + '\",\n')
            f.write('   \"pos\":' + str(pos) + '\n')
            f.write('}')
        with open(RUN_DATA_PATH + file_name,'w') as f:
            f.write('{\n')
            f.write('   \"weight_connect\":' + str(data['wc'].tolist()) + ',\n')
            f.write('   \"weight_in\":' + str(data['wi'].tolist()) + ',\n')
            f.write('   \"weight_out\":' + str(data['wo'].tolist()) + ',\n')
            f.write('   \"biase_connect\":' + str(data['bc'].tolist()) + ',\n')
            f.write('   \"biase_in\":' + str(data['bi'].tolist()) + ',\n')
            f.write('   \"biase_out\":' + str(data['bo'].tolist()) + ',\n')
            c,h = tuple(data['st'])
            f.write('   \"c\":' + str(c.tolist()) + ',\n')
            f.write('   \"h\":' + str(h.tolist()) + '\n')
            f.write('}')

    def __loadData(self):
        with open(RUN_DATA_RECORDER,'r') as load_f:
            load_dict = json.load(load_f)
        with open(RUN_DATA_PATH + load_dict['newest_data'],'r') as load_f:
            load_data_dict = json.load(load_f)
        load_data_dict['pos'] = load_dict['pos']
        return load_data_dict

    def __trainOne(self,one_times,learning_rate):
        """
        训练一张图
        """
        # 当前图像id
        img_id = self.reader.get_next_pic_id()
        print('>.<)o  training No.' + str(img_id) + ' pic')
        img_batch = tf.Variable(numpy.array(self.reader.get_pic_data(img_id)),dtype = tf.float32)
        img_batch = tf.reshape(img_batch,[1,224,224,3])
        # 获取VGG19网络得到的结果
        data = self.__getVGG19Result(img_batch)
        # 全连接层向量化VGG19隐藏层输出的4096维数据 -> 256维
        data_vgg = tf.cast(data,tf.float32)
        data_vgg = tf.reshape(data_vgg,(1,4096))
        data0 = tf.add(tf.matmul(data_vgg, self.weights['connect']), self.biases['connect']) 
        # 添加激活函数
        data0 = tf.nn.relu(data0)
        # data0.shape = (1,256)
        qa = self.reader.get_pic_qa(img_id)
        for q in qa:
            question = self.dealer.deal(q['question'])
            # print(q['answers'])
            answers = []
            confidences = []
            for a in q['answers']:
                answer = self.dealer.deal(a['answer'])
                # maybe代表着十成正确
                confidence = 1 
                answers.append(answer)
                confidences.append(confidence)
            # 创建one_hot常量标签
            label_batch = self.__getOnehot(answers,confidences,self.dealer.getWordNum())
            label_batch = tf.cast(label_batch,tf.float32)
            label_batch = tf.reshape(label_batch,(1,self.dealer.getWordNum()))
            data = []
            for word in question:
                data.append(self.__getOnehot(0,word,256))
            data = tf.cast(data,tf.float32)
            # print(data0)
            # print(data)
            data = tf.concat([data0,data],0)
            # data.shape = (-1,256)
            data = tf.add(tf.matmul(data, self.weights['in']), self.biases['in']) 
            data = tf.reshape(data,[1,len(question) + 1,256])
            data = tf.nn.relu(data)
            # 进入LSTM网络训练
            # 隐藏层层数:len(question) + 1
            # outputs为最后一层的输出(1,len(question) + 1,256)
            # 默认第一个隐藏层参数为0
            outputs, states = tf.nn.dynamic_rnn(self.lstm_cell, data, initial_state=self.init_state, time_major=False, dtype = tf.float32)
            # print(states) # states包含了C与H的值
            outputs = tf.reshape(outputs,(1,len(question) + 1,256,1))
            # 平均池化
            output = tf.nn.avg_pool(value=outputs, ksize=[1, len(question) + 1, 1, 1], strides=[1, len(question) + 1, 1, 1], padding='VALID')
            output = tf.reshape(output,(1,256))
            result = tf.add(tf.matmul(output, self.weights['out']), self.biases['out'])
            result = tf.sigmoid(result)
            # 防止梯度爆炸
            # result = tf.clip_by_value(result,1e-8,1.0-1e-8)
            # 计算交叉熵
            # -(tf.log(result)*label_batch + (1 - label_batch)*tf.log(1 - result))
            loss = tf.reduce_mean(tf.abs(result - label_batch))
            # 计算准确率
            accuracy = tf.abs(result - label_batch)
            # print(accuracy)
            accuracy = 1 - tf.reduce_mean(accuracy)
            # 建立优化器 随机梯度下降
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
            # 减少误差，提升准确度
            train = optimizer.minimize(loss)
            # 输入所有的变量
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                coord = tf.train.Coordinator()  
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for m in range(one_times):
                    sess.run(train)
                    ac,re = sess.run([accuracy,result])
                    # print(re)
                    print('accuracy:',end = '')
                    print(ac)
                weight_connect,weight_in,weight_out = sess.run([self.weights['connect'],self.weights['in'],self.weights['out']])
                biase_connect,biase_in,biase_out = sess.run([self.biases['connect'],self.biases['in'],self.biases['out']])
                st = sess.run(states)
                data_run =  {'wc':weight_connect,'wi':weight_in,'wo':weight_out,'bc':biase_connect,'bi':biase_in,'bo':biase_out,'st':st}
                try:
                    # 请求线程终止
                    coord.request_stop()
                except tf.errors.OutOfRangeError:  
                    print ('Done training -- epoch limit reached') 
                finally:
                    # 请求线程终止
                    coord.request_stop()
                    coord.join(threads)
        # 返回数据
        return data_run


    def train(self,times = 1,one_times = 1,learning_rate = 10,save_mod = 10):
        """
        训练网络
        times:次数
        one_times:一次训练次数(一个batch)
        learning_rate:学习率
        save_mod:每隔save_mod张图保存一次
        """
        for i in range(times):
            info = psutil.virtual_memory()
            print ("memory use:"+str(psutil.Process(os.getpid()).memory_info().rss))
            print ("memory all:"+str(info.total))
            print ("memory %:"+str(info.percent))
            data_run = self.__trainOne(one_times,learning_rate)
            if i % save_mod == 0:
                self.__saveData(data_run,self.reader.get_pos())
        if not i % save_mod == 0:
            self.__saveData(data_run,self.readei6r.get_pos())

if __name__ == '__main__':
    net = TrainNetForVQA()
    net.train(times = 10,one_times = 10,save_mod = 2)