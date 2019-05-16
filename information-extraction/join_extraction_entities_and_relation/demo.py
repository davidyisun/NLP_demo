#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 关系实体的联合抽取
Created on 2019-01-23
@author:David Yisun
@group:data
"""

# import paddle.fluid as fluid
# import paddle.v2 as paddle
# from paddle.fluid.initializer import NormalInitializer
# from paddle.v2.plot import Ploter
import re
import math
import os

import json
import numpy as np


train_title = "Train cost"
test_title = "Test cost"
# plot_cost = Ploter(train_title, test_title)
step = 0

# =============================================global parameters and hyperparameters==================================
EMBEDDING = 300
DROPOUT = 0.5
LSTM_ENCODE = 300
LSTM_DECODE = 600
BIAS_ALPHA = 10
VALIDATION_SIZE = 0.1
FILE_PATH = 'D:\\workspace_for_python\\to_github\\data\\model\\information_extract\\NYT'
TRAIN_PATH = os.path.join(FILE_PATH,  'train.json')
TEST_PATH = os.path.join(FILE_PATH,  'test.json')
X_TRAIN = os.path.join(FILE_PATH,  'sentence_train.txt')
Y_TRAIN = os.path.join(FILE_PATH,  'seq_train.txt')
X_TEST = os.path.join(FILE_PATH,  'sentence_test.txt')
Y_TEST = os.path.join(FILE_PATH,  'seq_test.txt')
WORD_DICT = os.path.join(FILE_PATH,  'word_dict.txt')
TAG_DICT = os.path.join(FILE_PATH,  'tag_dict.txt')
EPOCH_NUM = 1000
BATCH_SIZE = 128


# =============================================get data from the dataset==============================================
def get_data(train_path, test_path, train_valid_size):
    '''
    extracting data for json file
    '''
    train_file = open(train_path).readlines()
    x_train = []
    y_train = []
    for i in train_file:
        data = json.loads(i)
        x_data, y_data = data_decoding(data)
        '''
        appending each single data into the x_train/y_train sets
        '''
        x_train += x_data
        y_train += y_data

    test_file = open(test_path).readlines()
    x_test = []
    y_test = []
    for j in test_file:
        data = json.loads(j)
        x_data, y_data = data_decoding(data)
        x_test += x_data
        y_test += y_data
    return x_train, y_train, x_test, y_test


def data_decoding(data):
    '''
    decode the json file
    sentText is the sentence
    each sentence may have multiple types of relations
    for every single data, it contains: (sentence-splited, labels)
    '''
    sentence = data["sentText"]
    relations = data["relationMentions"]
    x_data = []
    y_data = []
    for i in relations:
        entity_1 = i["em1Text"].split(" ")
        entity_2 = i["em2Text"].split(" ")
        relation = i["label"]
        relation_label_1 = entity_label_construction(entity_1)
        relation_label_2 = entity_label_construction(entity_2)
        output_list = sentence_label_construction(sentence, relation_label_1, relation_label_2, relation)
        x_data.append(sentence.split(" "))
        y_data.append(output_list)
    return x_data, y_data


def entity_label_construction(entity):
    '''
    give each word in an entity the label
    for entity with multiple words, it should follow the BIES rule
    '''
    relation_label = {}
    for i in range(len(entity)):
        if i == 0 and len(entity) >= 1:
            relation_label[entity[i]] = "B"
        if i != 0 and len(entity) >= 1 and i != len(entity) - 1:
            relation_label[entity[i]] = "I"
        if i == len(entity) - 1 and len(entity) >= 1:
            relation_label[entity[i]] = "E"
        if i == 0 and len(entity) == 1:
            relation_label[entity[i]] = "S"
    return relation_label


def sentence_label_construction(sentence, relation_label_1, relation_label_2, relation):
    '''
    combine the label for each word in each entity with the relation
    and then combine the relation-entity label with the position of the entity in the triplet
    '''
    element_list = sentence.split(" ")
    dlist_1 = list(relation_label_1)
    dlist_2 = list(relation_label_2)
    output_list = []
    for i in element_list:
        if i in dlist_1:
            output_list.append(relation + '-' + relation_label_1[i] + '-1')
        elif i in dlist_2:
            output_list.append(relation + '-' + relation_label_2[i] + '-2')
        else:
            output_list.append('O')
    return output_list


def format_control(string):
    str1 = re.sub(r'\r', '', string)
    str2 = re.sub(r'\n', '', str1)
    str3 = re.sub(r'\s*', '', str2)
    return str3


# def joint_extraction():
#     vocab_size = len(open(WORD_DICT, 'r').readlines())
#     tag_num = len(open(TAG_DICT, 'r').readlines())
#
#     def bilstm_lstm(word, target, vocab_size, tag_num):
#         x = fluid.layers.embedding(
#             input=word,
#             size=[vocab_size, EMBEDDING],
#             dtype="float32",
#             is_sparse=True)
#
#         y = fluid.layers.embedding(
#             input=target,
#             size=[tag_num, tag_num],
#             dtype="float32",
#             is_sparse=True)
#
#         fw, _ = fluid.layers.dynamic_lstm(
#             input=fluid.layers.fc(size=LSTM_ENCODE * 4, input=x),
#             size=LSTM_ENCODE * 4,
#             candidate_activation="tanh",
#             gate_activation="sigmoid",
#             cell_activation="sigmoid",
#             bias_attr=fluid.ParamAttr(
#                 initializer=NormalInitializer(loc=0.0, scale=1.0)),
#             is_reverse=False)
#
#         bw, _ = fluid.layers.dynamic_lstm(
#             input=fluid.layers.fc(size=LSTM_ENCODE * 4, input=x),
#             size=LSTM_ENCODE * 4,
#             candidate_activation="tanh",
#             gate_activation="sigmoid",
#             cell_activation="sigmoid",
#             bias_attr=fluid.ParamAttr(
#                 initializer=NormalInitializer(loc=0.0, scale=1.0)),
#             is_reverse=True)
#
#         combine = fluid.layers.concat([fw, bw], axis=1)
#
#         decode, _ = fluid.layers.dynamic_lstm(
#             input=fluid.layers.fc(size=LSTM_DECODE * 4, input=combine),
#             size=LSTM_DECODE * 4,
#             candidate_activation="tanh",
#             gate_activation="sigmoid",
#             cell_activation="sigmoid",
#             bias_attr=fluid.ParamAttr(
#                 initializer=NormalInitializer(loc=0.0, scale=1.0)),
#             is_reverse=False)
#
#         softmax_connect = fluid.layers.fc(input=decode, size=tag_num)
#
#         _cost = fluid.layers.softmax_with_cross_entropy(
#             logits=softmax_connect,
#             label=y,
#             soft_label=True)
#         _loss = fluid.layers.mean(x=_cost)
#         return _loss, softmax_connect
#
#     source = fluid.layers.data(name="source", shape=[1], dtype="int64", lod_level=1)
#     target = fluid.layers.data(name="target", shape=[1], dtype="int64", lod_level=1)
#
#     loss, softmax_connect = bilstm_lstm(source, target, vocab_size, tag_num)
#     return loss


def get_index(word_dict, tag_dict, x_data, y_data):
    x_out = [word_dict[str(k)] for k in x_data]
    y_out = [tag_dict[str(l)] for l in y_data]
    return [x_out, y_out]


def data2index(WORD_DICT, TAG_DICT, x_train, y_train):
    def _out_dict(word_dict_path, tag_dict_path):
        word_dict = {}
        f = open(word_dict_path, 'r').readlines()
        for i, j in enumerate(f):
            word = re.sub(r'\n', '', str(j))
            #             word = re.sub(r'\r','',str(j))
            #             word = re.sub(r'\s*','',str(j))
            word_dict[word] = i + 1

        tag_dict = {}
        f = open(tag_dict_path, 'r').readlines()
        for m, n in enumerate(f):
            tag = re.sub(r'\n', '', str(n))
            tag_dict[tag] = m + 1
        return word_dict, tag_dict

    def _out_data():
        word_dict, tag_dict = _out_dict(WORD_DICT, TAG_DICT)
        for data in list(zip(x_train, y_train)):
            x_out, y_out = get_index(word_dict, tag_dict, data[0], data[1])
            yield x_out, y_out

    return _out_data


# def optimizer_program():
#     return fluid.optimizer.Adam()


if __name__ == "__main__":
    sentence_train, seq_train, sentence_test, seq_test = get_data(TRAIN_PATH, TEST_PATH, VALIDATION_SIZE)

    # train_reader = paddle.batch(
    #     paddle.reader.shuffle(
    #         data2index(WORD_DICT, TAG_DICT, sentence_train, seq_train),
    #         buf_size=500),
    #     batch_size=128)
    #
    # test_reader = paddle.batch(
    #     paddle.reader.shuffle(
    #         data2index(WORD_DICT, TAG_DICT, sentence_test, seq_test),
    #         buf_size=500),
    #     batch_size=128)
    #
    # place = fluid.CPUPlace()
    # feed_order = ['source', 'target']
    # trainer = fluid.Trainer(
    #     train_func=joint_extraction,
    #     place=place,
    #     optimizer_func=optimizer_program)
    #
    # trainer.train(
    #     reader=train_reader,
    #     num_epochs=100,
    #     event_handler=event_handler_plot,
    #     feed_order=feed_order)

