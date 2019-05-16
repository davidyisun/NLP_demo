#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: fasttext 预测模型
    过程:
    process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
    登录 --> 建立对话 --> 喂数据 --> 预测
Created on 2019-01-27
@author:David Yisun
@group:data
"""
# 配置gpu资源
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'  # 使用 GPU 0
os.path.abspath('../')

import tensorflow as tf
import numpy as np
from .fastText_model_multilabel import fastTextB as fastText
from utils.data_util import load_data_predict, load_final_test_data, create_voabulary, create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
import codecs

#configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size", 1999, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled", 100, "number of noise sampling")
tf.app.flags.DEFINE_string("ckpt_dir", "./model", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 300, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", False, "is traniing.true:training,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 15, "epoch number")
tf.app.flags.DEFINE_integer("validate_every", 3, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_string("predict_target_file", "fast_text_checkpoint_multi/zhihu_result_ftB_multilabel.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file", 'test-zhihu-forpredict-v4only-title.txt',"target file path for final prediction")

# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_voabulary()

