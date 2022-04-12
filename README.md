# Multi-label enhancement based self-supervised deep cross-modal hashing (MESDCH)

Pytorch implementation of paper 'Multi-label enhancement based self-supervised deep cross-modal
hashing'.

## Abstract

Deep cross-modal hashing which integrates deep learning and hashing into cross-modal retrieval,
achieves better performance than traditional cross-modal retrieval methods. Nevertheless, most previous
deep cross-modal hashing methods only utilize single-class labels to compute the semantic affinity
across modalities but overlook the existence of multiple category labels, which can capture the semantic
affinity more accurately. Additionally, almost all existing cross-modal hashing methods straightforwardly
employ all modalities to learn hash functions but neglect the fact that original instances in all modalities
may contain noise. To avoid the above weaknesses, in this paper, a novel multi-label enhancement based
self-supervised deep cross-modal hashing (MESDCH) approach is proposed. MESDCH first propose a
multi-label semantic affinity preserving module, which uses ReLU transformation to unify the similarities
of learned hash representations and the corresponding multi-label semantic affinity of original instances
and defines a positive-constraint Kullback–Leibler loss function to preserve their similarity. Then this
module is integrated into a self-supervised semantic generation module to further enhance the performance
of deep cross-modal hashing. Extensive evaluation experiments on four well-known datasets
demonstrate that the proposed MESDCH achieves state-of-the-art performance and outperforms several
excellent baseline methods in the application of cross-modal hashing retrieval.

------

Please cite our paper if you use this code in your own work:

@article{zou2022multi,
  title={Multi-label enhancement based self-supervised deep cross-modal hashing},
  author={Zou, Xitao and Wu, Song and Bakker, Erwin M and Wang, Xinzhi},
  journal={Neurocomputing},
  volume={467},
  pages={138--162},
  year={2022},
  publisher={Elsevier}
}

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+
----

### Dataset

we implement our method on dataset Mirflckr25K:

(1) please download the original image-text data of Mirflckr25K from http://press.liacs.nl/mirflickr/mirdownload.html  and put it under the folder /dataset/data/.

(2) please download the mirflickr25k-fall.mat, mirflickr25k-iall.mat, mirflickr25k-lall.mat and mirflickr25k-yall.mat from https://pan.baidu.com/s/1FX82NhdtnTeARcgmqxYCag 
(提取码：imk4) and put them under the folder /dataset/data/.

### How to run
 
 Step1: Run function run in main.py

If you have any problems, please feel free to contact Xitao Zou (xitaozou@mail.swu.edu.cn).
