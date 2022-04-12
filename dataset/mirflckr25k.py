# -*- coding: utf-8 -*-
# @Time    : 2019/6/27
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .base import *
from scipy import io as sio
import numpy as np

all_num = 20015
img_names = None
txt = None
label = None

img_dir = '/home/lab1008/zxt/datasets/MIRFLICKR25K/mirflickr/'
imgname_mat_url = '/home/lab1008/zxt/datasets/MIRFLICKR25K/FAll/mirflickr25k-fall.mat'
tag_mat_url = '/home/lab1008/zxt/datasets/MIRFLICKR25K/YAll/mirflickr25k-yall.mat'
label_mat_url = '/home/lab1008/zxt/datasets/MIRFLICKR25K/LAll/mirflickr25k-lall.mat'

def load_mat(img_mat_url: str, tag_mat_url: str, label_mat_url: str):
    img_names = sio.loadmat(img_mat_url)['FAll']  # type: np.ndarray
    img_names = img_names.squeeze()
    all_img_names = np.array([name[0] for name in img_names])
    all_txt = np.array(sio.loadmat(tag_mat_url)['YAll'])
    all_label = np.array(sio.loadmat(label_mat_url)['LAll'])
    return all_img_names, all_txt, all_label


def split_data(all_img_names, all_txt, all_label, query_num=2000, train_num=10000, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(range(20015))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_img_names = all_img_names[query_index]
    train_img_names = all_img_names[train_index]
    retrieval_img_names = all_img_names[retrieval_index]

    query_txt = all_txt[query_index]
    train_txt = all_txt[train_index]
    retrieval_txt = all_txt[retrieval_index]

    query_label = all_label[query_index]
    train_label = all_label[train_index]
    retrieval_label = all_label[retrieval_index]

    img_names = (query_img_names, train_img_names, retrieval_img_names)
    txt = (query_txt, train_txt, retrieval_txt)
    label = (query_label, train_label, retrieval_label)
    return img_names, txt, label


def get_single_datasets(img_dir=img_dir, img_mat_url=imgname_mat_url, tag_mat_url=tag_mat_url, label_mat_url=label_mat_url, batch_size=128, train_num=10000, query_num=2000, seed=6):
    global img_names, txt, label
    if img_names is None:
        all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
        img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
        print("Mirflckr25K data load and shuffle by seed %d" % seed)
    print("load data set single Mirflckr25K")
    train_data = CrossModalSingleTrain(img_dir, img_names[1], txt[1], label[1], train_transform, batch_size)
    valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], txt[0], txt[2], label[0], label[2], valid_transform)
    return train_data, valid_data


def get_pairwise_datasets(img_dir=img_dir, img_mat_url=imgname_mat_url, tag_mat_url=tag_mat_url, label_mat_url=label_mat_url, batch_size=128, train_num=10000, query_num=2000, seed=6):
    global img_names, txt, label
    if img_names is None:
        all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
        img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
        print("Mirflckr25K data load and shuffle by seed %d" % seed)
    print("load data set pairwise Mirflckr25K")
    train_data = CrossModalPairwiseTrain(img_dir, img_names[1], txt[1], label[1], train_transform, batch_size)
    valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], txt[0], txt[2], label[0], label[2], valid_transform)
    return train_data, valid_data


def get_triplet_datasets(img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=128, train_num=10000, query_num=2000, seed=None):
    global img_names, txt, label
    if img_names is None:
        all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
        img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
        print("Mirflckr25K data load and shuffle by seed %d" % seed)
    print("load data set triplet Mirflckr25K")
    train_data = CrossModalTripletTrain(img_dir, img_names[1], txt[1], label[1], train_transform, batch_size)
    valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], txt[0], txt[2], label[0], label[2], valid_transform)
    return train_data, valid_data


__all__ = ['get_single_datasets', 'get_pairwise_datasets']
