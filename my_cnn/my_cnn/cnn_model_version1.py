import tensorflow as tf
import numpy as np
import time
import os
from python_dealdata import anchor_target_layer
##########################################################################
#  embedding_lookup + cnn + cosine margine ,  batch
##########################################################################
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
class InsQACNN(object):
    def __init__(self, sequence_length, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        :param sequence_length: 51
        :param batch_size: 30
        :param vocab_size: 语料的长度，包括问题和答案
        :param embedding_size: 100
        :param filter_sizes: 1,2,3,5
        :param num_filters: 400
        :param l2_reg_lambda: 0
        """
        #用户问题,字向量使用embedding_lookup
        self.input_x_1 = tf.placeholder(tf.float32, [batch_size, sequence_length,embedding_size,1], name="input_x_1")
        #待匹配正向问题
        self.input_x_2 = tf.placeholder(tf.float32, [batch_size, sequence_length,embedding_size,1], name="input_x_2")
        #负向问题
        self.input_y_3 = tf.placeholder(tf.float32, [batch_size,], name="input_y_3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(-1, trainable=False)
        l2_loss = tf.constant(0.0)
        print("input_x_1 ", self.input_x_1)

        # Embedding layer
        # with tf.device('/gpu:1'), tf.name_scope("embedding"):#整个过程是对输入以及输出还有随机提取的答案进行权重初始化
        #     W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W") #初始化原始输入对应embedding层权重，如果这样，对应的vector2就没有用到
        #     chars_1 = tf.nn.embedding_lookup(W, self.input_x_1)  #根据input_x_1 的大小以及index 在W中寻找对应是的元素 可以认为是不同词语的随机权重,在刚开始，以后的W会进行训练每一个词的w维度1*embedding
        #     chars_2 = tf.nn.embedding_lookup(W, self.input_x_2)  #根据input_x_1 的大小以及index 在W中寻找对应是的元素 可以认为是不同词语的随机权重
        #     chars_3 = tf.nn.embedding_lookup(W, self.input_y_3)  #根据input_x_1 的大小以及index 在W中寻找对应是的元素 可以认为是不同词语的随机权重
        #     self.embedded_chars_1 = tf.nn.dropout(chars_1, self.dropout_keep_prob) #对词语的权重随机进行drop抛弃
        #     self.embedded_chars_2 = tf.nn.dropout(chars_2, self.dropout_keep_prob)
        #     self.embedded_chars_3 = tf.nn.dropout(chars_3, self.dropout_keep_prob)
        #     self.embedded_chars_1 = chars_1 #batch_size*squence_length 语料固定，对应词语的权重在这里，每一个词大小embedding_size
        #     self.embedded_chars_2 = chars_2
        #     self.embedded_chars_3 = chars_3
        # self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1)#把它转化为一个一列，代表一个句子训练数据
        # self.embedded_chars_expanded_2 = tf.expand_dims(self.embedded_chars_2, -1)
        # self.embedded_chars_expanded_3 = tf.expand_dims(self.embedded_chars_3, -1)

        pooled_outputs_1 = []#卷积核跨越词多少不同的变化，输出不同
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        for i, filter_size in enumerate(filter_sizes): #filter_size=[1,2,3,5] #不同卷积核的大小
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size,1, num_filters] #num_filter=400 卷积核的大小 filter_size 表示卷积核跨越词的个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_x_1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
                )
                pooled_outputs_1.append(pooled)

                conv = tf.nn.conv2d(
                    self.input_x_2, #对应正确答案的卷积
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
                )
                pooled_outputs_2.append(pooled)

                # conv = tf.nn.conv2d(
                #     self.embedded_chars_expanded_3,
                #     W,
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="conv-3"
                # )
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="poll-3"
        #         )
        #         pooled_outputs_3.append(pooled)
        #pooled_outputs_3 #维度大小为[4,30,1,1,400] ,第一个为filter_size共有4个跨越词的个数30为batch_size
        num_filters_total = num_filters * len(filter_sizes)
        pooled_reshape_1 = tf.reshape(tf.concat(pooled_outputs_1, axis=3),[-1,
                                                                           ])  # 把对应的矩阵转化为一行，batch_size为行数[30,1600],就是为了更好地求一个句子的相似度q
        pooled_reshape_2 = tf.reshape(tf.concat(pooled_outputs_2, axis=3),[-1, num_filters_total])
        # pooled_reshape_3 = tf.reshape(tf.concat(pooled_outputs_3,axis=3), [-1, num_filters_total])
        pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)  # 这个是最终的向量，可以在这里进行sigmod(q*W*a)
        self.pooled_flat_1=pooled_flat_1
        pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        pooled_flat_2=tf.transpose(pooled_flat_2,)
        similar_weight=tf.Variable(tf.truncated_normal([num_filters_total,num_filters_total], stddev=0.1), name="W")
        print(similar_weight)
        # c = tf.matmul(self.pooled_flat_1, similar_weight)
        # self.final_result=tf.matmul(c,pooled_flat_2)
        b_final = tf.Variable(tf.constant(0.1), name="b")
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) #利用余弦相似度求解
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        pooled_len_3 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_3, pooled_flat_3), 1))#利用余弦相似度求解
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) #计算向量的点乘Batch模式
        pooled_mul_13 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_3), 1)

        with tf.name_scope("output"):
            self.cos_12 = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") #最后相似度的结果
            # self.cos_13 = tf.div(pooled_mul_13, tf.multiply(pooled_len_1, pooled_len_3)) #与不是正确答案进行相似度求解
        self.loss=-tf.reduce_mean(self.input_y_3*tf.log(tf.clip_by_value(self.cos_12,1e-10,1.0))+(1-self.input_y_3)*tf.log(tf.clip_by_value(1-self.cos_12,1e-10,1.0)))
        # self.loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.cos_12,logits=self.input_y_3)
        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        # margin = tf.constant(0.05, shape=[batch_size], dtype=tf.float32)
        # self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.cos_12, self.cos_13)))
        # self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
        optimizer = tf.train.AdamOptimizer(1e-4)
        # optimizer = tf.train.GradientDescentOptimizer(1e-2)  # 从tensorflow中获得优化器
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        #
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
        #summary
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                # grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                     tf.nn.zero_fraction(g))
                # grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(self.out_dir))

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
