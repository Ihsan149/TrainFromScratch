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

#constants go here
IMAGE_IN_PATH = '/home/09959295800/SERPRO/ProjetoEstrategico/ReconhecimentoImagem/CompCars/data/image/'
IMAGE_OUT_PATH = '/home/09959295800/SERPRO/ProjetoEstrategico/ReconhecimentoImagem/CompCars196/'
LOGS_PATH = './log/'
RANGE = 196
SPLIT_FOLDER = 'data/image/'

#functions go here
def main():
    create_dir_if_not_exists(IMAGE_OUT_PATH)

    log = setup_logger('logging', 'createsample.log', IMAGE_OUT_PATH + LOGS_PATH)
    log.info("Main-> Inicio Preparar Imagens para Treinamento.")
    log.info("Main-> Select a set of 196 images and generate train.txt and test.txt.")
    try:
        options = []
        for make in os.listdir(IMAGE_IN_PATH):
            for model in os.listdir(IMAGE_IN_PATH+make):
                options.append(IMAGE_IN_PATH+make+'/'+model)
        selected = []        
        for i in range(RANGE):
            element = choice(options)
            selected.append(element)
            options.pop(options.index(element))
        count_selected = len(set(selected))
        log.info('Counting selected: '+str(count_selected))
        listfolders = []
        for rootpath in selected:
            log.info('Reading path '+rootpath)
            for folder, subfolders, files in os.walk(rootpath):
                for file in files:
                    subfolder = folder.split(SPLIT_FOLDER)[-1]
                    class_folder = subfolder.split('/')[0]+'/'+subfolder.split('/')[1]
                    listfolders.append(class_folder)
                    outpath = IMAGE_OUT_PATH+SPLIT_FOLDER+subfolder
                    create_dir_if_not_exists(outpath)
                    file_original_path = os.path.join(folder, file)
                    file_destiny_path = os.path.join(outpath,file)
                    if os.path.exists(file_original_path):
                        copy2(file_original_path, file_destiny_path)
        print 'Counting saved: '+str(len(set(listfolders)))
    
    except Exception as e:
        log.exception("Unexpected error:")
        log.exception(e)

    log.info('Generation finished!!!')

    
#main program goes here
if __name__ == '__main__':
    main()
