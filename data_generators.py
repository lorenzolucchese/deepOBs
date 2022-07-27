import tensorflow as tf
import pandas as pd
import numpy as np

def CustomtfDataset(files, 
                    NF, 
                    horizon, 
                    n_horizons,
                    model_inputs = "orderbooks",
                    task = "classification", 
                    alphas = np.array([]), 
                    multihorizon = False, 
                    normalise = False,
                    teacher_forcing = False, 
                    window = 100, 
                    batch_size = 256, 
                    roll_window = 1):
    """
    :param dir: directory of files
    :param files: list of files in directory to use
    :param NF: number of features
    :param horizon: prediction horizon, 0, 1, 2, 3, 4 
    :param task: regression or classification
    :param alphas: array of alphas for class boundaries if task = classification.
    :param multihorizon: whether the predictions are multihorizon, if True overrides horizon
                         In this case trainX is [trainX, decoder]
    :param samples_per_file: how many samples are in each file
    :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder
    """
    # methods to be used
    def scale_fn(x, y):
        return x / tf.keras.backend.max(x), y

    def add_decoder_input(x, y):
        if teacher_forcing:
            if task == "classification":
                first_decoder_input = tf.keras.utils.to_categorical(tf.zeros(x.shape[0]), y.shape[-1])
                first_decoder_input = tf.reshape(first_decoder_input, [first_decoder_input.shape[0], 1, y.shape[-1]])
                decoder_input_data = tf.hstack((x[:, :-1, :], first_decoder_input))
            elif task == "regression":
                raise ValueError('teacher forcing with regression not yet implemented.')
            else:
                raise ValueError('task must be either classification or regression.')

        if not teacher_forcing:
            if task == "classification":
                # this sets the initial hidden state of the decoder to be y_0 = [0, 0, 0] for classification
                decoder_input_data = tf.zeros_like(y[:, 0:1, :])
            elif task == "regression":
                # this sets the initial hidden state of the decoder to be y_0 = 0 for regression
                decoder_input_data = tf.zeros_like(y[:, 0:1])
            else:
                raise ValueError('task must be either classification or regression.')

        return {'input': x, 'decoder_input': decoder_input_data}, y
    
    if multihorizon:
        horizon = slice(0, n_horizons)

    if (task == "classification")&(alphas.size == 0):
        raise ValueError('alphas must be assigned if task is classification.')

    # create combined dataset
    tf_datasets = []
    for file in files:
        if model_inputs in ["orderbooks", "orderflows"]:
            dataset = pd.read_csv(file).to_numpy()

            features = dataset[:, :NF]
            features = np.expand_dims(features, axis=-1)
            responses = dataset[(window-1):, NF:]
            responses = responses[:, horizon]

        elif model_inputs[:7] == "volumes":
            dataset = np.load(file)

            features = dataset['features']
            if model_inputs == "volumes":
                features = np.sum(features, axis = 2)
            D = features.shape[1]
            features = features[:, (D//2 - NF//2):(D//2 + NF//2)]
            features = np.expand_dims(features, axis=-1)
            features[features > 65504] = 65504
            features = tf.convert_to_tensor(features, dtype=tf.int16)
            
            responses = dataset['responses'][(window-1):, horizon]

        if task == "classification":
            if multihorizon:
                all_label = []
                for h in range(n_horizons):
                    one_label = (+1)*(responses[:, h]>=-alphas[h]) + (+1)*(responses[:, h]>alphas[h])
                    one_label = tf.keras.utils.to_categorical(one_label, 3)
                    one_label = one_label.reshape(len(one_label), 1, 3)
                    all_label.append(one_label)
                y = np.hstack(all_label)
            else:
                y = (+1)*(responses>=-alphas[horizon]) + (+1)*(responses>alphas[horizon])
                y = tf.keras.utils.to_categorical(y, 3)

        tf_datasets.append(tf.keras.preprocessing.timeseries_dataset_from_array(features, y, window, batch_size=batch_size, sequence_stride=roll_window))
    
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_datasets).flat_map(lambda x: x)  

    # # create combined dataset
    # features_list = []
    # y_list = []
    # for file in files:
    #     if model_inputs in ["orderbooks", "orderflows"]:
    #         dataset = pd.read_csv(file).to_numpy()

    #         features = dataset[:, :NF]
    #         features = tf.convert_to_tensor(np.expand_dims(features, axis=-1))
    #         responses = dataset[:, -n_horizons:]
    #         responses = responses[:, horizon]

    #     elif model_inputs[:7] == "volumes":
    #         dataset = np.load(file)

    #         features = dataset['features']
    #         if model_inputs == "volumes":
    #             features = np.sum(features, axis = 2)
    #         features = np.expand_dims(features, axis=-1)
    #         features[features > 65504] = 65504
    #         features = tf.convert_to_tensor(features, dtype=tf.int16)
            
    #         responses = dataset['responses'][:, horizon]

    #     if task == "classification":
    #         if multihorizon:
    #             all_label = []
    #             for h in range(responses.shape[1]):
    #                 one_label = (+1)*(responses[:, h]>=-alphas[h]) + (+1)*(responses[:, h]>alphas[h])
    #                 one_label = tf.keras.utils.to_categorical(one_label, 3)
    #                 one_label = one_label.reshape(len(one_label), 1, 3)
    #                 all_label.append(one_label)
    #             y = np.hstack(all_label)
    #         else:
    #             y = (+1)*(responses>=-alphas[horizon]) + (+1)*(responses>alphas[horizon])
    #             y = tf.keras.utils.to_categorical(y, 3)

    #     y = tf.convert_to_tensor(y)
    #     features_list.append(features)
    #     y_list.append(y)
    
    # features = tf.concat(features_list, axis=0)
    # y = tf.concat(y_list, axis=0)[(window-1):, ...]
    # tf_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(features, y, window, batch_size=batch_size, sequence_stride=roll_window)
    
    if normalise:
        tf_dataset = tf_dataset.map(scale_fn)

    if multihorizon:
        tf_dataset = tf_dataset.map(add_decoder_input)

    return tf_dataset