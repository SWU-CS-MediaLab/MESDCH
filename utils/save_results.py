import scipy.io as sio
from utils.valid import valid
import numpy as np
import os
from results.resultpath import result_dir


def save_hashcodes(batch_size, use_gpu, bit, img_model, txt_model, dataset_name, valid_data, method_name):
    print('starting to calculate and save hash codes for query set and retrieval set:')
    mAPi2t, mAPt2i, qB_img, qB_txt, rB_img, rB_txt, queryLabel, retrievalLabel = valid(batch_size, bit, use_gpu, img_model, txt_model,
                                                                                       valid_data, return_hash=True)
    qB_img = np.array(qB_img.numpy() > 0, dtype=np.float)
    qB_txt = np.array(qB_txt.numpy() > 0, dtype=np.float)
    rB_img = np.array(rB_img.numpy() > 0, dtype=np.float)
    rB_txt = np.array(rB_txt.numpy() > 0, dtype=np.float)
    queryLabel = np.array(queryLabel.numpy(), dtype=np.float)
    retrievalLabel = np.array(retrievalLabel.numpy(), dtype=np.float)
    sio.savemat(
        result_dir + "/hashCodes/" + method_name + "_" + str(bit) + "_" + dataset_name + ".mat",
        mdict={'q_img': qB_img, 'q_txt': qB_txt, 'r_img': rB_img, 'r_txt': rB_txt, 'queryLabel': queryLabel,
               'retrievalLabel': retrievalLabel})
    print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mAPi2t, mAPt2i))
    print('save hash codes finish!')
