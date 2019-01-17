#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: fasttext 训练
    process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
    fast text. using: very simple model;n-gram to captrue location information;h-softmax to speed up training/inference
Created on 2019-01-17
@author:David Yisun
@group:data
"""

import tensorflow as tf
import numpy as np
import os
import word2vec
import pickle
import h5py


#configuration
FLAGS=tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("cache_file_h5py", "/home/huzhiling/demo/data/text-classify/text-classification/data/ieee_zhihu_cup/data.h5","path of training/validation/test data.")  #../data/sample_multiple_label.txt 多标签文件
tf.app.flags.DEFINE_string("cache_file_pickle", "/home/huzhiling/demo/data/text-classify/text-classification/data/ieee_zhihu_cup/vocab_label.pik","path of vocabulary and label files")   #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 512批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.")  # 批处理的大小 32-->128 多少steps后开始衰减
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")  # 0.5一次衰减多少  衰减比例

tf.app.flags.DEFINE_integer("num_sampled",10,"number of noise sampling")  # 100  噪声取样
tf.app.flags.DEFINE_string("ckpt_dir", "fast_text_checkpoint_multi/", "checkpoint location for the model")  # 模型保存位置

