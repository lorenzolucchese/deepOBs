import tensorflow as tf
import pandas as pd
import numpy as np

def CustomtfDataset(files, NF, horizon, n_horizons, tot_horizons, model_inputs, task, alphas, multihorizon, normalise, batch_size, T, roll_window, shuffle, teacher_forcing = False):
    """
    Create custom tf.dataset object to be used by model.
    :param files: list of files to use
    :param NF: number of features
    :param horizon: prediction horizon
    :param n_horizons: number of horizons in multihorizon, slice(0, n_horizons)
    :param tot_horizons: total number of horizons present in each file
    :param model_inputs: which input is being used "orderbooks", "orderflows", "volumes" pr "volumes_L3"
    :param task: ML task, "regression" or "classification"
    :param alphas: (tot_horizons) array of alphas to use when task = "classification", (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty))
    :param multihorizon: whether the predictions are multihorizon, if True horizon = slice(0, n_horizons)
    :param normalise: whether to normalise the data
    :param T: length of lookback window for features
    :param batch_size: batch size for dataset
    :param roll_window: length of window to roll forward when extracting features/responses
    :param shuffle: boolean, whether to shuffle dataset
    :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder
    :return: tf_dataset: a tf.dataset object 
    """
    # methods to be used
    def scale_fn(x, y):
        if tf.keras.backend.max(x) == 0:
            x = tf.zeros_like(x, dtype=tf.float32)
        else:
            x = x / tf.keras.backend.max(x)
        return x, y

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
            responses = dataset[(T-1):, -tot_horizons:]
            responses = responses[:, horizon]

        elif model_inputs[:7] == "volumes":
            dataset = np.load(file)

            features = dataset['features']
            if model_inputs == "volumes":
                features = np.sum(features, axis = 2)
            mid = features.shape[1]
            features = features[:, (mid//2 - NF//2):(mid//2 + NF//2)]
            features = np.expand_dims(features, axis=-1)
            features = tf.convert_to_tensor(features, dtype=tf.float32)
            
            responses = dataset['responses'][(T-1):, horizon]

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

        tf_datasets.append(tf.keras.preprocessing.timeseries_dataset_from_array(features, y, T, batch_size=None, sequence_stride=roll_window, shuffle=False))
    
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_datasets).flat_map(lambda x: x)
    
    if normalise:
        tf_dataset = tf_dataset.map(scale_fn)

    if multihorizon:
        tf_dataset = tf_dataset.map(add_decoder_input)

    if shuffle:
        tf_dataset = tf_dataset.shuffle(1000, reshuffle_each_iteration=False)
    else:
        tf_dataset = tf_dataset.shuffle(1, reshuffle_each_iteration=False)

    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset

def CustomtfDatasetUniv(dict_of_files, NF, horizon, n_horizons, tot_horizons, model_inputs, task, dict_of_alphas, multihorizon, normalise, batch_size, T, roll_window, shuffle, teacher_forcing = False):
    """
    Create custom tf.dataset object to be used by model, when using multiple TICKERs with different files and alphas.
    :param dict_of_files: a dict containing the lists of files to use for each TICKER
    :param NF: number of features
    :param horizon: prediction horizon
    :param n_horizons: number of horizons in multihorizon, slice(0, n_horizons)
    :param tot_horizons: total number of horizons present in each file
    :param model_inputs: which input is being used "orderbooks", "orderflows", "volumes" pr "volumes_L3"
    :param task: ML task, "regression" or "classification"
    :param alphas: a dict containing the (tot_horizons,) arrays of alphas to use for each TICKER when task = "classification", (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty))
    :param multihorizon: whether the predictions are multihorizon, if True horizon = slice(0, n_horizons)
    :param normalise: whether to normalise the data
    :param T: length of lookback window for features
    :param batch_size: batch size for dataset
    :param roll_window: length of window to roll forward when extracting features/responses
    :param shuffle: boolean, whether to shuffle dataset
    :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder
    :return: tf_dataset: a tf.dataset object 
    """
    tf_datasets = []
    for TICKER in sorted(dict_of_files.keys()):
        files = dict_of_files[TICKER]
        alphas = dict_of_alphas[TICKER]
        tf_datasets.append(CustomtfDataset(files, 
                                           NF, 
                                           horizon, 
                                           n_horizons,
                                           tot_horizons,
                                           model_inputs = model_inputs,
                                           task = task, 
                                           alphas = alphas, 
                                           multihorizon = multihorizon, 
                                           normalise = normalise,
                                           teacher_forcing = teacher_forcing, 
                                           T = T, 
                                           batch_size = batch_size, 
                                           roll_window = roll_window,
                                           shuffle = shuffle))

    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_datasets).flat_map(lambda x: x)  

    return tf_dataset

def load_evalY(eval_files, horizon, n_horizons, tot_horizons, model_inputs, task, alphas, multihorizon, T, roll_window):
    """
    Load evalY, for debugging. For a given set of files load the corresponding responses.
    Note if comparing with a previously created tf.dataset object, this must not be shuffled.
    :param eval_files: a list of files with returns to use for evaluation
    :param horizon: prediction horizon
    :param n_horizons: number of horizons in multihorizon, slice(0, n_horizons)
    :param tot_horizons: total number of horizons present in each file
    :param model_inputs: which input is being used "orderbooks", "orderflows", "volumes" pr "volumes_L3"
    :param task: ML task, "regression" or "classification"
    :param alphas: a dict containing the (tot_horizons,) arrays of alphas to use for each TICKER when task = "classification", (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty))
    :param multihorizon: whether the predictions are multihorizon, if True horizon = slice(0, n_horizons)
    :param T: length of lookback window for features
    :param roll_window: length of window to roll forward when extracting features/responses
    :return: evalY: responses for evaluation, if multihorizon (:, n_horizons, 3), otherwise (:, 3)
    """
    evalY = np.array([])
    if multihorizon:
        evalY = evalY.reshape(0, n_horizons)
    for file in eval_files:
        if model_inputs in ["orderbooks", "orderflows"]:
            data = pd.read_csv(file).to_numpy()
            responses = data[(T-1)::roll_window, -tot_horizons:]
            responses = responses[:, horizon]
        elif model_inputs[:7] == "volumes":
            data = np.load(file)
            responses = data["responses"]
            responses = responses[(T-1)::roll_window, horizon]
        evalY = np.concatenate([evalY, responses])

    if task == "classification":
        if multihorizon:
            all_label = []
            for h in range(evalY.shape[1]):
                one_label = (+1)*(evalY[:, h]>=-alphas[h]) + (+1)*(evalY[:, h]>alphas[h])
                one_label = tf.keras.utils.to_categorical(one_label, 3)
                one_label = one_label.reshape(len(one_label), 1, 3)
                all_label.append(one_label)
            evalY = np.hstack(all_label)
        else:
            evalY = (+1)*(evalY>=-alphas[horizon]) + (+1)*(evalY>alphas[horizon])
            evalY = tf.keras.utils.to_categorical(evalY, 3)
    
    return evalY