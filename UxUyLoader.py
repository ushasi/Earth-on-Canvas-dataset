# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:33:06 2018

@author: ushasi2
"""
#from graphcnn.helper import *
import scipy.io
import numpy as np
import datetime
import h5py
#import graphcnn.setup.helper
#import graphcnn.setup as setup

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


def load_uxuy_dataset():

    datasetX = scipy.io.loadmat('data/Imgmat.mat')
    photos = np.squeeze(datasetX['feature'])  
    datasetY = scipy.io.loadmat('data/sketchmat.mat')
    sketches = np.squeeze(datasetY['feature'])  
    label_im = scipy.io.loadmat('data/softlabels.mat')
    label_im = np.squeeze(label_im['labels'])  
    label_sk = scipy.io.loadmat('data/softlabels.mat')
    label_sk = np.squeeze(label_sk['labels'])  
    #data1 = scipy.io.loadmat('dataset/names_images.mat')
    #data1 = np.squeeze(data1['data'])  
    #data2 = scipy.io.loadmat('dataset/names_sketches.mat')
    #data2 = np.squeeze(data2['data3'])
    #order = scipy.io.loadmat('dataset/dataset2.mat')
    #order = np.squeeze(order['datamat2'])  
    dataset = scipy.io.loadmat('data/wv_embeddings.mat')
    wv = np.squeeze(dataset['features'])  # word2vec features multi-label
    #dataset = scipy.io.loadmat('dataset/mst_graph.mat')
    #graph = np.squeeze(dataset['edge'])  # word2vec features multi-label

    np.array(photos) 
    np.array(sketches) 
    np.array(label_im)
    np.array(label_sk) 
    #np.array(data1)
    #np.array(data2) 
    #np.array(order)
    np.array(wv)
    #np.array(graph) 
    label_im = label_im.flatten()
    label_sk = label_sk.flatten()
    #data1 = data1.flatten()
    #data2 = data2.flatten()
    np.array(wv) 
    print("Training set (images X) shape: {shape}", photos.shape)
    print("Training set (sketches Y) shape: {shape}", sketches.shape) 
    print("Training set (labels images) shape: {shape}", label_im.shape)
    print("Training set (labels sketches) shape: {shape}", label_sk.shape) 
    print("Training set (wv) shape: {shape}", wv.shape)     
    #loading features in which NaN values have been replaced
      

   
    return photos, sketches, label_im, label_sk, wv
'''
scipy.io
h5py
scipy
sklearn
tensorflow gpu
'''
