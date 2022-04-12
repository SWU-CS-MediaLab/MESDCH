import torch


def calc_hammingDist(B1, B2):
    q = B2.shape[1]  # length of bit, e.g. 64
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))  # 返回18015个数，返回值越小，表示相似性越大
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]#查询样本总数
    map = 0
    if k is None:
        k = retrieval_L.shape[0]#验证集样本总数
    for iter in range(num_query):  # 2000
        q_L = query_L[iter]  # 待查询样本标签
        if len(q_L.shape) < 2:  # 标签如果是一维的,24，变为二维1*24
            q_L = q_L.unsqueeze(0)
        q_L=q_L.float()
        retrieval_L=retrieval_L.float()
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)  # mm:#矩阵乘; 这行返回18015个数，表示查询样本的标记与query_L中每一个样本标记的乘积,然后变成1（相似）或者0（不相似）
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)  # 1*18015
        _, ind = torch.sort(hamm)  # 升序
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))  # 标记相似的数量，也即原始数据中相似样本的数量
        count = torch.arange(1, total + 1).type(torch.float32)  # top 1、top 2、top 3 。。。
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0  # torch.nonzero(gnd)输出张量中的每行包含gnd中非零元素的索引;这句难理解!!!
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    map = calc_map_k(qB, rB, query_L, retrieval_L)
    print(map)
