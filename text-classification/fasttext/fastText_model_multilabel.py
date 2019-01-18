#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: fasttest 核心模型
Created on 2019-01-17
@author:David Yisun
@group:data
"""

import tensorflow as tf

class fastTextB(object):
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate, num_sampled, sentence_len, vocab_size,embed_size,is_training,max_label_per_example=5):
        """init all hyperparameter here"""
        # 1.set hyper-paramter
        self.label_size = label_size  # e.g.1999
        self.batch_size = batch_size  # batch size
        self.num_sampled = num_sampled  # 样本数
        self.sentence_len = sentence_len  # 样本长度
        self.vocab_size = vocab_size  # 词表大小
        self.embed_size = embed_size  # embedding 维度
        self.is_training = is_training  # 是否训练
        self.learning_rate = learning_rate  # 学习率
        self.max_label_per_example = max_label_per_example  # 每个样本的label最大个数
        self.initializer = tf.random_normal_initializer(stddev=0.1)  # 正太初始化

        # 2.add placeholder (X,label)
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")     # X
        self.labels = tf.placeholder(tf.int64, [None, self.max_label_per_example], name="Labels")   # y [1,2,3,3,3]
        self.labels_l1999 = tf.placeholder(tf.float32,[None, self.label_size])  # int64

        # 3.set some variables
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")   #
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")

        # 4.init weights 初始化权重
        self.instantiate_weights()

        # 5.main_graph: inference
        self.logits = self.inference() # [None, self.label_size]

        # 6.calculate loss
        self.loss_val = self.loss()

        # 7.start training by update parameters using according loss
        self.train_op = self.train()





    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size],
                                         initializer=self.initializer)
        self.W = tf.get_variable("W", [self.embed_size, self.label_size], initializer=self.initializer)
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get embedding of words in the sentence
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)  # [None,self.sentence_len,self.embed_size]

        # 2.average vectors, to get representation of the sentence  映射出句子向量
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)  # [None,self.embed_size]

        # 3.linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b   # [None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        return logits

    def loss(self, l2_lambda=0.0001):

        labels_multi_hot = self.labels_l1999  # [batch_size,label_size]
        # sigmoid_cross_entropy_with_logits:Computes sigmoid cross entropy given `logits`.
        # Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.
        # For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_multi_hot, logits=self.logits)  # labels:[batch_size, label_size], logits:[batch_size, label_size]
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))  # 算总的loss
        print('loss:{0}'.format(loss))

        # add regularization result in not converge
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda  # ？所有训练变量的l2 loss？
        print("l2_losses:", self.l2_losses)
        loss = loss + self.l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)  # 学习率衰减
        # train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=self.loss_val, global_step=self.global_step)
        return train_op






