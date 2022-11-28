# deepOBs
This repo contains the code for the paper ['The Short-Term Predictability of Returns in Order Book Markets: A Deep Learning Perspective'](https://arxiv.org/abs/2211.13777).

The main methods, which we believe could be of use to other researchers, are found in:
- [_data_process.py_](https://github.com/lorenzolucchese/deepOBs/blob/master/data_process.py): functions for processing order book data dowloaded from [LOBSTER](https://lobsterdata.com/) to raw order book, order flow and volume features and the corresponding returns;
- [_data_methods.py_](https://github.com/lorenzolucchese/deepOBs/blob/master/data_methods.py): auxiliary functions for processed data;
- [_custom_datasets.py_](https://github.com/lorenzolucchese/deepOBs/blob/master/custom_datasets.py): create custom _tf.dataset_ objects to load features and responses into models;
- [_model.py_](https://github.com/lorenzolucchese/deepOBs/blob/master/model.py): a class to build, train and evaluate deepLOB (Zhang et al., [2019](https://ieeexplore.ieee.org/document/8673598)), deepOF (Kolm et al., [2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141)) and deepVOL as _keras.models.Model_ objects;
- [_MCS_results.py_](https://github.com/lorenzolucchese/deepOBs/blob/master/MCS_results.py): functions to perform the bootstrap Model Confidence Set (Hansen et al., [2011](https://www.jstor.org/stable/41057463)) procedure on results

![alt text](https://github.com/lorenzolucchese/deepOBs/blob/master/auxiliary_code/core.png)
