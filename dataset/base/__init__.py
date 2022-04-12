# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
from __future__ import absolute_import
from __future__ import print_function

from .base import CrossModalTrainBase, CrossModalValidBase, train_transform, valid_transform
from .pairwise import CrossModalPairwiseTrain
from .single import CrossModalSingleTrain
from .triplet import CrossModalTripletTrain
from .quadruplet import CrossModalQuadrupletTrain


__all__ = ['CrossModalTrainBase', 'CrossModalValidBase', 'CrossModalSingleTrain',
           'CrossModalPairwiseTrain', 'CrossModalTripletTrain', 'CrossModalQuadrupletTrain', 'train_transform', 'valid_transform']
