from torch import nn
from utils.utils import calc_map_k
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def valid(batch_size, bit, use_gpu, img_model: nn.Module, txt_model: nn.Module, dataset, return_hash=False):
    # get query img and txt binary code
    dataset.query()
    qB_img = get_img_code(batch_size, bit, use_gpu, img_model,  dataset)
    qB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    query_label = dataset.get_all_label()
    # get retrieval img and txt binary code
    dataset.retrieval()
    rB_img = get_img_code(batch_size, bit, use_gpu, img_model, dataset)
    rB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    retrieval_label = dataset.get_all_label()
    mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
    mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)
    if return_hash:
        return mAPi2t, mAPt2i, qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu(), query_label, retrieval_label
    return mAPi2t, mAPt2i

def get_img_code(batch_size, bit, use_gpu, img_model: nn.Module, dataset, isPrint=False):
    dataset.img_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True, pin_memory=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    B_img = torch.zeros(len(dataset), bit, dtype=torch.float)
    if use_gpu:
        B_img = B_img.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        img = data['img']  # type: torch.Tensor
        if use_gpu:
            img = img.cuda()
        f = img_model(img)
        B_img[index, :] = f.data
        if isPrint:
            print(B_img[index, :])
    B_img = torch.sign(B_img)
    return B_img.cpu()


def get_txt_code(batch_size, bit, use_gpu, txt_model: nn.Module, dataset, isPrint=False):
    dataset.txt_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, drop_last=True, pin_memory=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    B_txt = torch.zeros(len(dataset), bit, dtype=torch.float)
    if use_gpu:
        B_txt = B_txt.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        txt = data['txt']  # type: torch.Tensor
        txt = txt.float()

        if use_gpu:
            txt = txt.cuda()
        g = txt_model(txt)
        B_txt[index, :] = g.data
        if isPrint:
            print(B_txt[index, :])
    B_txt = torch.sign(B_txt)
    return B_txt.cpu()
