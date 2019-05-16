#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: textcnn 训练
    process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict

Created on 2019-01-30
@author:David Yisun
@group:data
"""
# 配置gpu资源
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'  # 使用 GPU 0
import numpy as np
from .TextCNN_model import TextCNN
import pickle
import h5py
import os
import random

import tensorflow as tf


#configuration
FLAGS=tf.app.flags.FLAGS

# hd5格式的训练 验证 测试数据集
tf.app.flags.DEFINE_string("cache_file_h5py", "/data/demo/data/text-classify/text-classification/data/ieee_zhihu_cup/data.h5", "path of training/validation/test data.") #../data/sample_multiple_label.txt
# pickle 词表
tf.app.flags.DEFINE_string("cache_file_pickle", "/data/demo/data/text-classify/text-classification/data/ieee_zhihu_cup/vocab_label.pik", "path of vocabulary and label files") #../data/sample_multiple_label.txt
# 学习率 0.0003
tf.app.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
# batch_size 64
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
# 衰减步长 1000
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
# 衰减率 1
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
# 模型保存地址 ./model/
tf.app.flags.DEFINE_string("ckpt_dir", "./model/", "checkpoint location for the model")
# 句子长度 多少个词
tf.app.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
# embed_size 词向量维度 128
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
# 训练 还是 验证/预测 True
tf.app.flags.DEFINE_boolean("is_training_flag", True, "is training.true:tranining,false:testing/inference")
# 训练轮数
tf.app.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
# 每一轮做一次验证
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每1轮做一次验证
# 是否使用外部embedding
tf.app.flags.DEFINE_boolean("use_embedding", False, "whether to use embedding or not.")
# filter个数
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") # 256--->512
# w2v 地址
tf.app.flags.DEFINE_string("word2vec_model_path", "word2vec-title-desc.bin", "word2vec's vocabulary and vectors")
# 命名空间
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
# 是否是多标签
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
# filter size N*M
filter_sizes = [6, 7, 8]






