#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2019--
@author:David Yisun
@group:data
"""
def demo_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__file__)

    # 将日志写入到文件
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")

if __name__ == '__main__':
    demo_logging()