#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: trie树
Created on 2019-07-01
@author: David Yisun
@group: data
@e-mail: david_yisun@163.com
@describe:
"""

LETTER_NUM = 27  # 组成单词的字母个数，26个字母+'-'


# Trie 结构体
class Node:
    def __init__(self, is_word=False):
        global LETTER_NUM
        self.is_word = is_word  # 是不是单词结束节点
        self.prefix_count = 0  # 这个前缀的单词个数
        self.children = [None for child in range(LETTER_NUM)]


# Trie 结构体
class Trie:
    def __init__(self):
        self.head = Node()

    ###插入新单词
    def insert(self, word):
        current = self.head
        count = 0

        for letter in word:
            if (letter == '-'):
                int_letter = LETTER_NUM - 1
            else:
                int_letter = ord(letter) - ord('a')
            if (current.children[int_letter] is None):
                current.children[int_letter] = Node()
                current = current.children[int_letter]
                count += 1
                current.prefix_count = count
            else:
                current = current.children[int_letter]
                current.prefix_count += 1
        current.is_word = True

    ###查询单词是否存在
    def search(self, word):
        current = self.head
        int_letter = 0
        for letter in word:
            if (letter == '-'):
                int_letter = LETTER_NUM - 1
            else:
                int_letter = ord(letter) - ord('a')

            if (current.children[int_letter] is None):
                # print "int_letter = " + str(int_letter)
                return False
            else:
                current = current.children[int_letter]
        return current.is_word

    ###根据字母前缀输出所有的单词
    def output(self, strPrefix):
        if (strPrefix is None or strPrefix == ""):
            print("please tell me prefix letter.")
        currentNode = self.head
        int_letter = 0
        for letter in strPrefix:
            if (letter == '-'):
                int_letter = LETTER_NUM - 1
            else:
                int_letter = ord(letter) - ord('a')
            currentNode = currentNode.children[int_letter]

        if (currentNode is not None):
            if (currentNode.is_word):
                print(strPrefix + " ")
        else:
            return

        for i in range(LETTER_NUM):
            if (currentNode.children[i] is not None):
                self.output(strPrefix + chr(i + ord('a')))

        #################    


###读取单词列表文本构造Trie结构
class BuildTrie:

    def __init__(self):
        self.trie = Trie()
        for line in file("EnglishDict.txt"):
            line = line.lower()  # 全部换成小写
            line = line.replace('\r', '').replace('\n', '')  # 去掉结束符
            isword = True
            int_letter = 0
            str_letter = "abcdefghijklmnopqrstuvwxyz-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for letter in line:
                if (letter not in str_letter):
                    isword = False
                    break
            if (isword == False):
                print(line + ", it is not a word")
                continue
            else:
                self.trie.insert(line)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    #    t = Trie()
    #    t.insert("apple")
    #    t.insert("abc")
    #    t.insert("abandon")
    #    t.insert("bride")
    #    t.insert("bridegroom")
    #    t.insert("good")
    #    t.output("b")

    bt = BuildTrie()
    t = bt.trie
    t.output("z")

    print(t.search("apple"))
    print(t.search("fff"))
    print(t.search("good")
    print("a num:" + str(t.head.children[0].prefix_count))
    print("ab num:" + str(t.head.children[0].children[1].prefix_count))
    print("b num:" + str(t.head.children[1].prefix_count))
