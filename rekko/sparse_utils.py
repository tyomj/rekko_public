import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

np.random.seed = 0

def create_interaction_matrices(train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                dict_of_users: dict,
                                dict_of_items: dict,
                                dtype: type = np.float32,
                                logger = None) \
        -> (csr_matrix, csr_matrix):
    """
    Creates two interaction matrices for train and test puposes.

    train_df: train interactions DataFrame
    test_df: test interactions DataFrame
    dict_of_users: dictionary of users key: user_id
    dict_of_items: dictionary of items key: item_id
    logger: an instance of `pkg`.logger.Logger, optional

    Returns:
        - 'train_csr' CSR matrix with train interactions
        - 'test_csr' CSR matrix with test interactions
    """

    # construct df indexes
    user_indices_train = np.array(train_df.user_id.apply(lambda x: dict_of_users[x]))
    item_indices_train = np.array(train_df.item_id.apply(lambda x: dict_of_items[x]))
    user_indices_test = np.array(test_df.user_id.apply(lambda x: dict_of_users[x]))
    item_indices_test = np.array(test_df.item_id.apply(lambda x: dict_of_items[x]))

    # sanity
    if logger:
        if len(user_indices_train) != len(item_indices_train):
            logger.error('Num of indexes must be the same!')
        if len(user_indices_test) != len(item_indices_test):
            logger.error('Num of indexes must be the same!')

    # to numpy
    data_train = np.array(train_df.rating.astype(int))
    data_test = np.array(test_df.rating.astype(int))

    # store lens
    uc = len(dict_of_users)
    ic = len(dict_of_items)

    # create sparse
    train_csr = csr_matrix((np.array(data_train), (np.array(user_indices_train), np.array(item_indices_train))),
                           shape=(uc, ic), dtype = dtype)
    test_csr = csr_matrix((np.array(data_test), (np.array(user_indices_test), np.array(item_indices_test))),
                          shape=(uc, ic), dtype = dtype)

    # sanity
    if logger:
        if train_csr.shape != test_csr.shape:
            logger.error('Matrix shapes must be the same!')
    if logger:
        logger.info('Matrix filling train {0:.4f} %'.format(
            train_csr.count_nonzero() / (train_csr.shape[0] * train_csr.shape[1]) * 100))
        logger.info('Matrix filling test {0:.4f} %'.format(
            test_csr.count_nonzero() / (test_csr.shape[0] * test_csr.shape[1]) * 100))
        logger.info('Memory usage train {0:.2f} Mb'.format(train_csr.data.nbytes / 1024 / 1024))
        logger.info('Memory usage test {0:.2f} Mb'.format(test_csr.data.nbytes / 1024 / 1024))

    return train_csr, test_csr

def create_interaction_matrix(df: pd.DataFrame, 
                              unique_items: set, 
                              dtype: type = np.float32, 
                              logger = None) -> (csr_matrix, dict, dict):
    """
    Creates an interaction matrix.

    df: interactions DataFrame
    logger: an instance of `pkg`.logger.Logger, optional

    Returns:
        - 'df' DataFrame with train interactions
        - 'dict_of_users' dictionary of users key: user_id
        - 'dict_of_items' dictionary of items key: item_id
    """

    # list of ids
    list_of_users = [i for i in np.sort(df.user_id.unique())]
    #list_of_items = [i for i in np.sort(df.item_id.unique())]
    list_of_items = unique_items

    if logger:
        logger.info('Total num of users: {}'.format(len(list_of_users)))
        logger.info('Total num of items: {}'.format(len(list_of_items)))

    data = np.array(df.rating.astype(int))

    # index and id
    dict_of_users = dict(list(zip(list_of_users, range(len(list_of_users)))))
    dict_of_items = dict(list(zip(list_of_items, range(len(list_of_items)))))

    uc = len(dict_of_users)
    ic = len(dict_of_items)

    user_indices = np.array(df.user_id.apply(lambda x: dict_of_users[x]))
    item_indices = np.array(df.item_id.apply(lambda x: dict_of_items[x]))

    csr = csr_matrix((np.array(data), (np.array(user_indices), np.array(item_indices))),
                           shape=(uc, ic), dtype = dtype)
    if logger:
        logger.info('Matrix filling train {0:.4f} %'.format(
            csr.count_nonzero() / (csr.shape[0] * csr.shape[1]) * 100))
        logger.info('Memory usage train {0:.2f} Mb'.format(csr.data.nbytes / 1024 / 1024))

    return csr, dict_of_users, dict_of_items