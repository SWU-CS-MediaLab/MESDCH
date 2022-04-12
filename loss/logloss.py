import torch
def logloss(labels_batchsize, labels_train, hashrepresentations_batchsize, hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    S = (labels_batchsize.matmul(labels_train.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    theta = 1.0 / 2 * torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t())
    logloss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta))) / (batch_size * num_train)
    return logloss