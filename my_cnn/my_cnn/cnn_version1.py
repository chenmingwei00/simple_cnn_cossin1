import tensorflow as tf
import os
import datetime
import time
import numpy as np
import pickle
from cnn_model_version1 import InsQACNN
import insurance_qa_data_helpers
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
class cnn_version1():
    '''
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 3000, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps (default: 100)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    '''

    def __init__(self):
        self.dropout_keep_prob=1.0
        self.batch_size=10
        self.num_epochs=500000
        self.evaluate_every=3000
        self.checkpoint_every=3000
        self.embedding_dim=300
        self.num_filters=400
        self.filter_sizes="1,2,3,5"
        self.l2_reg_lambda=0.05
        self.allow_soft_placement=True
        self.log_device_placement=False
        self.get_data()
        f = open('E:/tianchi_tran/my_cnn/competation_data/spanish_vec.pkl', 'rb')
        self.words_vec = pickle.load(f)
        self.model=InsQACNN(
            sequence_length=70,  # in this it is 263 每一个句子长度
            batch_size=self.batch_size,
            vocab_size=len(self.vocab),
            embedding_size=self.embedding_dim,
            filter_sizes=list(map(int, self.filter_sizes.split(","))),
            num_filters=self.num_filters,
            l2_reg_lambda=self.l2_reg_lambda
        )
    def get_data(self):
        self.vocab,self.train_y,self.train_x_1,self.train_x_2 = insurance_qa_data_helpers.build_vocab()  # index of question'word and answer'words
        self.test_x1,self.test_x2=insurance_qa_data_helpers.build_vocab_test()
        # self.alist = insurance_qa_data_helpers.read_alist()  # get question'answer
        # self.raw = insurance_qa_data_helpers.read_raw()  # raw train set for example:1', 'qid:0', 'can_felon_have_Life_Insurance_<a>_<a>_<ave_a_felony_question_be_a_grade_death_benefit_life_insurance_policy_they_be_essentially_guarantee_issue_plan_with_low_death_benefit_
    def train(self):
        # 获取输入输出
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)

            sess.run(tf.global_variables_initializer())
            with sess.as_default():
                saver,checkpoint_prefix,train_summary_writer=self.summary_fun(sess=sess)

                for i in range(self.num_epochs):
                    try:
                        start = time.time()
                        self.x_train_1, self.x_train_2, self.y_train_3 ,self.words_vec= insurance_qa_data_helpers.load_data_temp(self.words_vec,self.train_x_1,self.train_x_2,self.train_y, self.batch_size)
                        self.x_train_1=np.array([self.x_train_1])
                        self.x_train_1=self.x_train_1.reshape(self.x_train_1.shape[1],self.x_train_1.shape[2],self.x_train_1.shape[3],1)
                        self.x_train_2 = np.array([self.x_train_2])
                        self.x_train_2 = self.x_train_2.reshape(self.x_train_2.shape[1], self.x_train_2.shape[2],
                                                               self.x_train_2.shape[3], 1)
                        self.y_train_3=self.y_train_3.astype(float)
                        # print(self.x_train_1.shape)
                        # print(self.train_y.shape)
                        self.train_step(self.x_train_1, self.x_train_2, self.y_train_3, sess=sess,train_summary_writer=train_summary_writer)#
                        end = time.time()
                        current_step = tf.train.global_step(sess, self.model.global_step)
                        if current_step % self.checkpoint_every == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                    except Exception as e:
                        print(e)
                    break

    def summary_fun(self,sess):
        # Define Training procedure
        train_summary_writer = tf.summary.FileWriter(self.model.dev_summary_dir, sess.graph_def)
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(self.model.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        return saver, checkpoint_prefix, train_summary_writer

    def train_step(self,x_batch_1, x_batch_2, y_batch_3,sess,train_summary_writer):#)
        """
        A single training step
        """
        feed_dict = {
            self.model.input_x_1: x_batch_1, #在这里是实际数据
            self.model.input_x_2: x_batch_2,
            self.model.input_y_3: y_batch_3,
            self.model.dropout_keep_prob: self.dropout_keep_prob
        }
        output1,output2=sess.run([self.model.pooled_flat_1,self.model.pooled_flat_2],feed_dict)
        print(np.array(output1).shape)
        print(np.array(output1))
        _, step, summaries, loss, accuracy = sess.run(
            [self.model.train_op, self.model.global_step, self.model.train_summary_op, self.model.loss, self.model.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        if step%100==0:
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def predict(self):
        with tf.Session() as sess:
            model_path = '/home/admin/chen/chen/tianchi_trans/trans_cnn/cnn_qa/insuranceQA-cnn-lstm-master/my_cnn/runs/1528569309/checkpoints/'
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path, "******************")
                tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
            else:
                print("没找到模型")

            final_result=[]
            test_length=len(self.test_x1)
            print(test_length)
            start_index=0
            while start_index<test_length:
                if start_index%500==0:
                    print("pre_:",start_index)
                self.test_x1_block,self.test_x2_block,self.words_vec=insurance_qa_data_helpers.load_data_temp_test(self.words_vec,self.test_x1,self.test_x2,start_index)# 包含问题，答案以及对应错误答案,其中每一个元素表示对应的词语下标
                self.test_x1_block = np.array([self.test_x1_block])
                self.test_x1_block = self.test_x1_block.reshape(self.test_x1_block.shape[1], self.test_x1_block.shape[2],
                                                        self.test_x1_block.shape[3], 1)
                self.test_x2_block = np.array([self.test_x2_block])
                self.test_x2_block = self.test_x2_block.reshape(self.test_x2_block.shape[1], self.test_x2_block.shape[2],
                                                        self.test_x2_block.shape[3], 1)

                feed_dict = {
                    self.model.input_x_1: self.test_x1_block,
                    self.model.input_x_2: self.test_x2_block,
                    self.model.dropout_keep_prob: self.dropout_keep_prob
                }
                coss_result = sess.run(self.model.cos_12, feed_dict)
                final_result.extend(coss_result)
                start_index+=10
            print(len(final_result))
            with open("/home/admin/chen/chen/tianchi_trans/trans_cnn/cnn_qa/insuranceQA-cnn-lstm-master/my_cnn/cnn_predict_data/cnn_pre_1.txt","w") as pre_1:
                for result_cur in final_result:
                    pre_1.write(str(result_cur))
                    pre_1.write('\n')



if __name__ == '__main__':
    cnn=cnn_version1()
    cnn.train()