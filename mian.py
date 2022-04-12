import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from training.MESDCH import train

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    datasets = ['mirflickr25k']
    bits = [16,32,64]
    for ds in datasets:
        for bit in bits:
            train(ds, bit, batch_size=128, issave=False, max_epoch=500)  

    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

