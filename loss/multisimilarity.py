import torch


def multilabelsimilarityloss_KL(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    KLloss = torch.sum(torch.mul(labelsSimilarity - hashrepresentationsSimilarity,
                                 torch.log((1e-5 + labelsSimilarity) / (1e-5 + hashrepresentationsSimilarity)))) / (
                         num_train * batch_size)
    # KLloss2 = torch.sum(torch.relu(labelsSimilarity - hashrepresentationsSimilarity)) / (num_train * batch_size)
    # KLloss3 = torch.sum(torch.relu(hashrepresentationsSimilarity - labelsSimilarity)) / (num_train * batch_size)
    # KLloss = KLloss1 + 0.5 * KLloss2 + 0.5 * KLloss3
    # print('KLloss1 = %4.4f, KLloss2 = %4.4f'%(KLloss1 , KLloss2))
    return KLloss


def multilabelsimilarityloss_MSE(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    MSEloss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (num_train * batch_size)

    return MSEloss
