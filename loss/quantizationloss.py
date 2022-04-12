import torch
import numpy as np

def quantizationLoss(hashrepresentations_bs, hashcodes_bs):
    batch_size, bit = hashcodes_bs.shape
    quantization_loss = torch.sum(torch.pow(hashcodes_bs - hashrepresentations_bs, 2)) / (batch_size * bit)
    return quantization_loss


