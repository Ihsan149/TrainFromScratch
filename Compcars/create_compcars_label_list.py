#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:56:21 2017

@author: 09959295800
"""
#imports go here
import os
from pyutils.misc import setup_logger 
from pyutils.misc import create_dir_if_not_exists

#constants go here
ROOT_PATH = '/home/09959295800/SERPRO/ProjetoEstrategico/ReconhecimentoImagem/CompCars196/'
IMAGE_IN_PATH = 'data/image/'
IMAGE_OUT_PATH = 'data/train_test_split/classification/'
LOGS_PATH = 'log/'
SPLIT_FOLDER = 'data/image/'


#functions go here
def main():

    log = setup_logger('logging', 'createlabels.log', ROOT_PATH + LOGS_PATH)
    log.info("Main-> Gerar listas de imagens para treinamento e teste com Caffe.")
    log.info("Main-> Create a set of images and generate train.txt and test.txt.")
    
    create_dir_if_not_exists(ROOT_PATH + IMAGE_OUT_PATH)
    
    try:
        class_pathes = []
        for folder, subfolders, files in os.walk(ROOT_PATH + IMAGE_IN_PATH):
            for file in files:
                tail = folder.split(SPLIT_FOLDER)[-1]
                class_path = tail.split('/')[0]+'/'+tail.split('/')[1]
                class_pathes.append(class_path)
        class_pathes = sorted(set(class_pathes))
        
        #saving to file...
        train_output = open(ROOT_PATH + IMAGE_OUT_PATH + 'class_labels.txt','w') 
        class_label = 0
        for path in class_pathes:
            train_output.write(str(path) + ' ' + str(class_label) + '\n')
            class_label += 1
        train_output.close()
    
    except Exception as e:
        log.exception("Unexpected error:") #, sys.exc_info()[0])
        log.exception(e)

    log.info('Generation finished!!!')

    
#main program goes here
if __name__ == '__main__':
    main()
