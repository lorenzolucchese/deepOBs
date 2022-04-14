import numpy as np
import tensorflow as tf
from keras.layers import Reshape, concatenate

if __name__ == '__main__':
    # file_path = "data/model/AAL_volumes_W1/test/test000.npz"
    # with np.load(file_path) as data:
    #     x = tf.convert_to_tensor(data["X"])
    #     y = tf.convert_to_tensor(data["Y"])
    # print("x.shape = ", x.shape)
    # print("y.shape = ", y.shape)

    # print("max(x) = ", np.max(x[245,:, :,0]))
    # print("y = ", y[0,:,:])

    NF = 40
    T = 2

    input_lmd = tf.constant(np.arange(NF*T), shape = [1,T, NF, 1])
    model = tf.keras.Sequential()
    print(input_lmd.shape)
    input_BID = Reshape((T, NF//2, 1, 1))(input_lmd[:, :, :NF//2, :])
    input_BID = tf.reverse(input_BID, axis = [2])
    input_ASK = Reshape((T, NF//2, 1, 1))(input_lmd[:, :, NF//2:, :])
    input_lmd = concatenate([input_BID, input_ASK], axis = 3)
    print(input_lmd.shape)
    print(input_lmd[0, 0, :, :, 0])
    print(input_lmd[0, 1, :, :, 0])