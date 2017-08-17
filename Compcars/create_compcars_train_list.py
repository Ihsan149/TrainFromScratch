#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:56:21 2017

@author: 09959295800
"""
#imports go here
from random import choice
import os
from shutil import copy2
from pyutils.misc import setup_logger 
from pyutils.misc import create_dir_if_not_exists
from sklearn.model_selection import train_test_split

#constants go here
ROOT_PATH = '/home/09959295800/SERPRO/ProjetoEstrategico/ReconhecimentoImagem/CompCars196/'
IMAGE_IN_PATH = 'data/image/'
IMAGE_OUT_PATH = 'data/train_test_split/classification/'
IMAGE_TRAIN_PATH = 'data/train/'
IMAGE_VAL_PATH = 'data/val/'
LOGS_PATH = 'log/'
TRAIN_SET_SIZE = 0.7 # in percentage
SPLIT_FOLDER = 'data/image/'

#functions go here
def define_labels():
    labels = {}
    label_input = open(ROOT_PATH + IMAGE_OUT_PATH + 'class_labels.txt','r')
    for line in label_input:
        label = line.split()
        labels[label[0]] = label[1]
    label_input.close()
    return labels

def split_train():
    train_input = open(ROOT_PATH + IMAGE_OUT_PATH + 'train.txt','r')
    images = []
    for line in train_input:
        #image_name = line.split()[0]
        images.append(line)
    train_input.close()
    
    X_train, X_test = train_test_split(images, test_size = 0.3, random_state = 0)

    train_output = open(ROOT_PATH + IMAGE_OUT_PATH + 'train.txt','w')
    for image in X_train:
        train_output.write(image)
    train_output.close()
    
    test_output = open(ROOT_PATH + IMAGE_OUT_PATH + 'test.txt','w')
    for temp in X_test:
        image = temp.split('/')[-1] 
        test_output.write(image)
    test_output.close()

def main():

    log = setup_logger('logging', 'createsample.log', ROOT_PATH + LOGS_PATH)
    log.info("Main-> Gerar listas de imagens para treinamento e teste com Caffe.")
    log.info("Main-> Create a set of images and generate train.txt and test.txt.")
    
    create_dir_if_not_exists(ROOT_PATH + IMAGE_OUT_PATH)
    
    try:
        class_labels = define_labels()

        train_output = open(ROOT_PATH + IMAGE_OUT_PATH + 'train.txt','w') 
        test_output = open(ROOT_PATH + IMAGE_OUT_PATH + 'val.txt','w')
        create_dir_if_not_exists(ROOT_PATH + IMAGE_VAL_PATH)
        for folder, subfolders, files in os.walk(ROOT_PATH + IMAGE_IN_PATH):
            images = []
            for file in files:
                images.append(folder + '/' + file)
            train_size = int(round(len(images) * TRAIN_SET_SIZE))
            for i in range(train_size):
                image = choice(images)
                tail = image.split(SPLIT_FOLDER)[-1]
                image_class = tail.split('/')[0] + '/' + tail.split('/')[1]
                label =  class_labels[image_class]
                train_output.write(image_class + '/' + tail.split('/')[-1] + ' ' + label + '\n')
                file_original_path = image
                file_destiny_path = ROOT_PATH + IMAGE_TRAIN_PATH + image_class + '/' + tail.split('/')[-1]
                create_dir_if_not_exists(ROOT_PATH + IMAGE_TRAIN_PATH + image_class)
                if os.path.exists(file_original_path):
                    copy2(file_original_path, file_destiny_path)
                images.pop(images.index(image))
            for image in images:
                tail = image.split(SPLIT_FOLDER)[-1]
                image_class = tail.split('/')[0] + '/' + tail.split('/')[1]
                label =  class_labels[image_class]
                #removing the image path for val data
                #test_output.write(image_class + '/' + tail.split('/')[-1] + ' ' + label + '\n')
                test_output.write(tail.split('/')[-1] + ' ' + label + '\n')
                file_original_path = image
                file_destiny_path = ROOT_PATH + IMAGE_VAL_PATH + tail.split('/')[-1]
                if os.path.exists(file_original_path):
                    copy2(file_original_path, file_destiny_path)
        train_output.close()
        test_output.close()
        
        split_train()
    
    except Exception as e:
        log.exception("Unexpected error:") #, sys.exc_info()[0])
        log.exception(e)

    log.info('Generation finished!!!')

    
#main program goes here
if __name__ == '__main__':
    main()
    #split_train()
    