'''
@Author: WANG Maonan
@Date: 2023-10-29 14:26:09
@Description: feature convert
@LastEditTime: 2023-10-29 14:26:09
'''
import numpy as np

def onehot(num, size):
    """ One-hot encoding
    """
    onehot_vec = np.zeros(size)
    if num < size:
        onehot_vec[num] = 1
    return onehot_vec.tolist()