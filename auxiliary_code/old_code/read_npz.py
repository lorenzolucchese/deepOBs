import numpy as np
import tensorflow as tf
from keras.layers import Reshape, concatenate

if __name__ == '__main__':
    file_path = "data/model/AAL_volumes_W1_2/train/train000.npz"
    with np.load(file_path) as data:
        x = tf.convert_to_tensor(data["X"])
        y_reg = tf.convert_to_tensor(data["Y_reg"])
        y_class = tf.convert_to_tensor(data["Y_class"])
    print("x.shape = ", x.shape)
    print("y_reg.shape = ", y_reg.shape)
    print("y_class.shape = ", y_class.shape)

    print("x = ", x[0,:, :,0])
    print("y_reg = ", y_reg[0,:,...])
    print("y_class = ", y_class[0,:,...])

    # NF = 40
    # T = 2

    # input_lmd = tf.constant(np.arange(NF*T), shape = [1,T, NF, 1])
    # model = tf.keras.Sequential()
    # print(input_lmd.shape)
    # input_BID = Reshape((T, NF//2, 1, 1))(input_lmd[:, :, :NF//2, :])
    # input_BID = tf.reverse(input_BID, axis = [2])
    # input_ASK = Reshape((T, NF//2, 1, 1))(input_lmd[:, :, NF//2:, :])
    # input_lmd = concatenate([input_BID, input_ASK], axis = 3)
    # print(input_lmd.shape)
    # print(input_lmd[0, 0, :, :, 0])
    # print(input_lmd[0, 1, :, :, 0])