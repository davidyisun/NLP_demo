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

# 配置gpu资源
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'  # 使用 GPU 0
import tensorflow as tf
import numpy as np
import word2vec
import pickle
import h5py
from fastText_model_multilabel import fastTextB as fastText



#configuration
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("cache_file_h5py", "/home/huzhiling/demo/data/text-classify/text-classification/data/ieee_zhihu_cup/data.h5", "path of training/validation/test data.")  #../data/sample_multiple_label.txt 多标签文件
tf.app.flags.DEFINE_string("cache_file_pickle", "/home/huzhiling/demo/data/text-classify/text-classification/data/ieee_zhihu_cup/vocab_label.pik", "path of vocabulary and label files")   #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 512批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.")  # 批处理的大小 32-->128 多少steps后开始衰减
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")  # 0.5一次衰减多少  衰减比例

tf.app.flags.DEFINE_integer("num_sampled", 10, "number of noise sampling")  # 100  噪声取样
tf.app.flags.DEFINE_string("ckpt_dir", "./model", "checkpoint location for the model")  # 模型保存位置
tf.app.flags.DEFINE_integer("sentence_len", 200, "max sentence length")  # 句子长度 多少个词
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")  # 128 embedding size
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")  # 训练还是推断
tf.app.flags.DEFINE_integer("num_epochs", 25, "embedding size")  # epoch 数量
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证

tf.app.flags.DEFINE_boolean("use_embedding", False, "whether to use embedding or not.")  # 是否使用外部embedding

# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)


def load_data(cache_file_h5py, cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:", list(f_data.keys()))
    train_X=f_data['train_X']  # np.array(
    print("train_X.shape:", train_X.shape)
    train_Y=f_data['train_Y']  # np.array(
    print("train_Y.shape:", train_Y.shape, ";")
    vaild_X=f_data['vaild_X']  # np.array(
    valid_Y=f_data['valid_Y']  # np.array(
    test_X=f_data['test_X']  # np.array(
    test_Y=f_data['test_Y']  # np.array(

    word2index, label2index = None, None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index = pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y

# 赋值外部embedding
def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, fast_text):
    print("using pre-trained word emebedding.started...")
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load('zhihu-word2vec-multilabel.bin-100', kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(fast_text.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
def do_eval(sess, fast_text, evalX, evalY, batch_size, vocabulary_index2word_label):  # evalY1999
    evalX = evalX[0:3000]
    evalY = evalY[0:3000]
    number_examples, labels = evalX.shape
    print("number_examples for validation:", number_examples)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    batch_size = 1
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples,batch_size)):
        evalY_batch = process_labels(evalY[start:end])   # 整理成 [类别x， 类别x， 类别y， 类别y， 类别z] 的形式
        curr_eval_loss, logit = sess.run([fast_text.loss_val, fast_text.logits],  # curr_eval_acc-->fast_text.accuracy
                                          feed_dict={fast_text.sentence: evalX[start:end],
                                                     fast_text.labels_l1999: evalY[start:end]}) #,fast_text.labels_l1999:evalY1999[start:end]
        # print("do_eval.logits_", logits_.shape)
        label_list_top5 = get_label_using_logits(logit[0], vocabulary_index2word_label)  # 选出概率最大的前5个label
        curr_eval_acc = calculate_accuracy(list(label_list_top5), evalY_batch[0], eval_counter)  # evalY[start:end][0]
        print('evalY_batch shape: {0} * {1}'.format(evalY_batch.shape))
        eval_loss, eval_counter, eval_acc = eval_loss+curr_eval_loss, eval_counter+1, eval_acc+curr_eval_acc

    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)

def process_labels(trainY_batch, require_size=5, number=None):
    """
    限制label的输出维度  即 top N
    process labels to get fixed size labels given a spense label
    :param trainY_batch:
    :return:
    """
    #print("###trainY_batch:",trainY_batch)
    num_examples, _ = trainY_batch.shape
    trainY_batch_result = np.zeros((num_examples, require_size), dtype=int)

    for index in range(num_examples):
        y_list_sparse = trainY_batch[index]
        y_list_dense = [i for i, label in enumerate(y_list_sparse) if int(label) == 1]  # 选出每个样本所有标签 label向量中为1的
        y_list = process_label_to_algin(y_list_dense, require_size=require_size)
        trainY_batch_result[index] = y_list
        if number is not None and number%30 == 0:
            pass
            #print("####0.y_list_sparse:",y_list_sparse)
            #print("####1.y_list_dense:",y_list_dense)
            #print("####2.y_list:",y_list) # 1.label_index: [315] ;2.y_list: [315, 315, 315, 315, 315] ;3.y_list: [0. 0. 0. ... 0. 0. 0.]
    if number is not None and number % 30 == 0:
        #print("###3trainY_batch_result:",trainY_batch_result)
        pass
    return trainY_batch_result

def process_label_to_algin(ys_list, require_size=5):
    """
    given a list of labels, process it to fixed size('require_size')
    :param ys_list: a list
    :return: a list
    """
    ys_list_result = [0 for x in range(require_size)]
    if len(ys_list) >= require_size:  # 超长
        ys_list_result = ys_list[0:require_size]
    else:  # 太短
       if len(ys_list) == 1:
           ys_list_result = [ys_list[0] for x in range(require_size)]
       elif len(ys_list) == 2:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[0], ys_list[1], ys_list[1]]
       elif len(ys_list) == 3:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[1], ys_list[1], ys_list[2]]
       elif len(ys_list) == 4:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[1], ys_list[2], ys_list[3]]
    return ys_list_result

#从logits中取出前五 get label using logits  返回list的index
def get_label_using_logits(logits, vocabulary_index2word_label, top_number=5):
    index_list = np.argsort(logits)[-top_number:]  # 排序返回index 默认从小到大
    index_list = index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels, eval_counter):
    if eval_counter<10:
        print("labels_predicted:", labels_predicted, " ;labels:",labels)
    count = 0
    label_dict = {x: x for x in labels}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    return count / len(labels)




def main(_):
    # 1.load_data
    # word2index 词表
    # label2index label id
    # trainX 训练集  batch * sentence_len 句子长度

    trainX, trainY, testX, testY = None, None, None, None
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)  #
    index2label = {v: k for k, v in label2index.items()}  # label的id
    vocab_size = len(word2index)   # 词表大小
    print("cnn_model.vocab_size:", vocab_size)
    num_classes = len(label2index)  # 类别大小
    print("num_classes:", num_classes)
    num_examples, FLAGS.sentence_len = trainX.shape  # 获取 input 的 shape
    print("num_examples of training:", num_examples, "    sentence_len:", FLAGS.sentence_len)

    # 2. create session
    config = tf.ConfigProto()    # tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Instantiate Model
        fast_text=fastText(num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.num_sampled,
                           FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(FLAGS.ckpt_dir, "checkpoint")):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding 是否用外部训练的embedding
                vocabulary_index2word = {v: k for k, v in word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, fast_text)

        curr_epoch = sess.run(fast_text.epoch_step)

        # 3.feed data & training
        number_of_training_data = len(trainX)  # 训练数据量
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                curr_loss, current_l2_loss, _ = sess.run([fast_text.loss_val, fast_text.l2_losses, fast_text.train_op],
                                                         feed_dict={fast_text.sentence: trainX[start: end],
                                                                    fast_text.labels_l1999: trainY[start: end]})

                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])  # 2d-array. each element slength is a 100.
                    print("train_Y_batch:",
                          trainY[start:end])  # a list,each element is a list.element:may be has 1,2,3,4,5 labels.

                loss, counter = loss + curr_loss, counter + 1  # acc+curr_acc loss累加 counter：统计每个epoch的batch数
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tL2 Loss:%.3f" % (epoch, counter, loss / float(counter), current_l2_loss))
                    # \tTrain Accuracy:%.3f--->,acc/float(counter)

                if start % (1000 * FLAGS.batch_size) == 0:  # 每1000个batch 做一次验证
                    eval_loss, eval_accuracy = do_eval(sess, fast_text, vaildX, vaildY, batch_size,
                                                       index2label)  # testY1999,eval_acc
                    print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_accuracy))
                    # ,\tValidation Accuracy: %.3f--->eval_acc

                    # save model to checkpoint 保存模型  每6000 * FLAGS.batch_size 保存一次
                    if start % (6000 * FLAGS.batch_size) == 0:
                        print("Going to save checkpoint.")
                        save_path = FLAGS.ckpt_dir + "model.ckpt"
                        saver.save(sess, save_path, global_step=epoch)  # fast_text.epoch_step

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)  # 记录epoch数

            # 4.validation
            print("epoch:", epoch, "validate_every:", FLAGS.validate_every, "validate or not:", (epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_accuracy = do_eval(sess, fast_text, vaildX, vaildY, batch_size,
                                                   index2label)  # testY1999,eval_acc
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_accuracy))  # ,\tValidation Accuracy: %.3f--->eval_acc
                # save model to checkpoint
                print("Going to save checkpoint.")
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)  # fast_text.epoch_step

            # 5.最后在测试集上做测试，并报告测试准确率 Test
            test_loss, test_acc = do_eval(sess, fast_text, testX, testY, batch_size, index2label)  # testY1999
            print('测试集\nloss:{loss}    acc:{acc}'.format(loss=test_loss, acc=test_acc))
        pass


if __name__ == "__main__":
    tf.app.run()