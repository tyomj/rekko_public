import itertools
import janus
import mlflow
import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from implicit.nearest_neighbours import CosineRecommender
from sklearn.preprocessing import minmax_scale

from rekko.logger import Logger
from rekko.coordinator import ExperimentCoordinator
from rekko.sparse_utils import create_interaction_matrices, create_interaction_matrix
from rekko.metric import metric
from rekko.data.make_dataset import create_pkls
create_pkls()

from data_helpers import train_test_split_on_date, prepare_df_with_interactions, \
                        to_universal_df_view, combine_interaction_types,\
                        recency_function, csr_to_dict


ec = ExperimentCoordinator()
config_recency = ec.config.join('config_recency.yaml').load()
c = ec.base_coordinator
logger = Logger(path='', name='train', filename='train.log')

def train(alpha_=None, beta_=None):

    logger.info('Running ...')

    # Load all types of interactions, movies catalogue and test users
    transactions = c.data_interim.join('transactions.pkl').load()
    bookmarks = c.data_interim.join('bookmarks.pkl').load()
    ratings = c.data_interim.join('ratings.pkl').load()
    catalogue = c.data_interim.join('catalogue.pkl').load()
    test_users = c.data_interim.join('catalogue.pkl').load()

    logger.info('Data loaded')

    # Train/test split
    transactions_train, transactions_test = train_test_split_on_date(transactions)
    bookmarks_train, bookmarks_test = train_test_split_on_date(bookmarks)
    ratings_train, ratings_test = train_test_split_on_date(ratings)

    logger.info('Train/test split completed')

    # Processing
    ratings_train = to_universal_df_view(ratings_train)
    bookmarks_train = to_universal_df_view(bookmarks_train)
    transactions_train = to_universal_df_view(transactions_train)
    # ... and combining into a single dataframe
    all_interaction_train = combine_interaction_types(ratings_train, bookmarks_train, transactions_train)

    # Processing
    ratings = prepare_df_with_interactions(to_universal_df_view(ratings))
    bookmarks = prepare_df_with_interactions(to_universal_df_view(bookmarks))
    transactions = prepare_df_with_interactions(to_universal_df_view(transactions))

    # ... and combining into a single dataframe
    all_interaction = pd.concat([ratings, bookmarks, transactions])
    all_interaction = all_interaction.reset_index(drop=True)
    all_interaction = all_interaction.drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)

    unique_items = set(catalogue.element_uid.unique())

    # Creating global csr matrix
    csr, dict_of_users, dict_of_items = create_interaction_matrix(all_interaction, unique_items)
    assert np.all([k == v for k, v in dict_of_items.items()])

    logger.info('Csr_matrix with all interactions created')

    interaction_train = prepare_df_with_interactions(transactions_train)
    interaction_test = prepare_df_with_interactions(transactions_test)
    real_int_train_csr, real_int_test_csr = create_interaction_matrices(interaction_train, interaction_test,
                                                                        dict_of_users, dict_of_items, logger=None)

    logger.info('Data preparation finished')

    # count true labels for validation
    # validation dictionary for train data and test data
    train_true_dict = csr_to_dict(real_int_train_csr)
    test_true_dict = csr_to_dict(real_int_test_csr)
    # and an example with set inside for time optimization during filtering
    train_true_dict_set = {k: set(v) for k, v in train_true_dict.items()}
    # test_true_dict_set = {k: set(v) for k, v in test_true_dict.items()}

    # ----------------------------------------------------------------------------------------- #
    # Recency function parameter search
    if (alpha_ is None) and (beta_ is None):

        # Prepare attributes for recency function
        all_interaction_train['time_scaled'] = minmax_scale(all_interaction_train.ts)
        # Все операции выполняем на трейне
        all_interaction_train = all_interaction_train.merge(
            all_interaction_train.groupby('element_uid').time_scaled.min().reset_index().rename(
                {'time_scaled': 'element_launch_ts'}, axis=1))
        all_interaction_train['seen_ts_since_launch'] = all_interaction_train['time_scaled'] - all_interaction_train[
            'element_launch_ts']

        all_interaction_train = all_interaction_train[
            ['element_uid', 'user_uid', 'element_launch_ts', 'seen_ts_since_launch']]

        # takes parameters to search
        alpha_par = config_recency['grid_params']['alpha']
        beta_par = config_recency['grid_params']['beta']

        alpha_range_params = np.arange(alpha_par['min'], alpha_par['max'], alpha_par['step'])
        beta_range_params = np.arange(beta_par['min'], beta_par['max'], beta_par['step'])

        iters = [alpha_range_params, beta_range_params]
        all_variants = list(itertools.product(*iters))
        np.random.shuffle(all_variants)

        logger.info('Starting search ...')

        # ----------------------------------------------------------------------------------------- #

        for element in all_variants:

            alpha_ = element[0]
            beta_ = element[1]

            interaction_train_ = recency_function(all_interaction_train, alpha_, beta_)
            train_csr, test_csr = create_interaction_matrices(interaction_train_, interaction_test, dict_of_users,
                                                              dict_of_items, logger=None)
            train_true_dict_set = {k: set(v) for k, v in train_true_dict.items()}

            model = CosineRecommender(K=10200)
            model.fit(train_csr.T, show_progress=False)
        
            # without filtering in model
            test_predict = {}
            for id_ in tqdm(np.unique(test_csr.nonzero()[0])):
                test_predict[id_] = model.recommend(id_, train_csr, N=300, filter_already_liked_items=False)
            test_predict = {k: [x[0] for x in v] for k, v in test_predict.items()}
            # get rid of movies watched in train
            test_predict = {k: [x for x in v if x not in train_true_dict_set.get(k, [])][:20]
                            for k, v in tqdm(test_predict.items())}

            mapk = metric.metric(test_true_dict, test_predict)
            logger.info('alpha = {0}, beta = {1}, mnap@20 = {2}'.format(alpha_, beta_, mapk))

            # dump mlflow params
            run = mlflow.start_run(experiment_id=0)
            mlflow.set_tag("tag", "Implicit_with_recency")
            mlflow.log_param('lib', 'implicit')
            mlflow.log_param('feedbacks_mode', 'implicit')
            mlflow.log_param('type', 'CF')
            # search-related params
            mlflow.log_param('alpha', alpha_)
            mlflow.log_param('beta', beta_)
            mlflow.log_metric('MNAP_at_20_test', mapk)
            mlflow.end_run()
        # ----------------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------------------------- #
    # Pruning parameter search
    else:
        pruning_range = config_recency['grid_params']['max_len']
        
        all_interaction_train = all_interaction_train.sort_values(by=['user_uid', 'ts'], ascending=False)
        for max_len in range(pruning_range['min'], pruning_range['max'], pruning_range['step']):
            # ----------------------------------------------------------------------------------------- #
            logger.info('max_len = {}'.format(max_len))
            all_interaction_train_pruned = all_interaction_train.groupby('user_uid').apply(lambda x: x[:max_len]).reset_index(drop=True)
        
            # Все операции выполняем на трейне
            all_interaction_train_pruned = all_interaction_train_pruned.merge(all_interaction_train_pruned.groupby('element_uid').time_scaled.min().reset_index().rename({'time_scaled': 'element_launch_ts'}, axis=1))
            all_interaction_train_pruned['seen_ts_since_launch'] = all_interaction_train_pruned['time_scaled'] - all_interaction_train_pruned['element_launch_ts']
            all_interaction_train_pruned = all_interaction_train_pruned[['element_uid', 'user_uid', 'element_launch_ts', 'seen_ts_since_launch']]

            inv_test_users = [dict_of_users.get(k, None) for k in test_users['users']]
            inv_test_users = [k for k in inv_test_users if k is not None]
            inv_test_users = set(inv_test_users)
            test_true_dict_50k = {k: v for k, v in test_true_dict.items() if k in inv_test_users}
            
            interaction_train_ = recency_function(all_interaction_train_pruned, int(alpha_), int(beta_))
            train_csr, test_csr = create_interaction_matrices(interaction_train_, interaction_test, dict_of_users, dict_of_items, logger=None)
            
            model = CosineRecommender(K=10200)
            model.fit(train_csr.T, show_progress=False)
            
            # without filtering in model
            test_predict = {}
            for id_ in tqdm(inv_test_users):
                test_predict[id_] = model.recommend(id_, train_csr, N=300, filter_already_liked_items=False)
            test_predict = {k: [x[0] for x in v] for k, v in test_predict.items()}
            # get rid of movies watched in train
            test_predict = {k: [x for x in v if x not in train_true_dict_set.get(k, [])][:20] for k, v in tqdm(test_predict.items())}
            mapk = metric.metric(test_true_dict_50k, test_predict)
            logger.info('mapk = {}'.format(mapk))

            # dump mlflow params
            run = mlflow.start_run(experiment_id=1)
            mlflow.set_tag("tag", "Implicit_with_pruning")
            mlflow.log_param('lib', 'implicit')
            mlflow.log_param('feedbacks_mode', 'implicit')
            mlflow.log_param('type', 'CF')
            # search-related params
            mlflow.log_param('max_len', max_len)
            mlflow.log_metric('MNAP_at_20_test', mapk)
            mlflow.end_run()
            # ----------------------------------------------------------------------------------------- #
            
if __name__ == '__main__':
    p = janus.ArgParser()
    p.new_str('a alpha')
    p.new_str('b beta')
    p.parse()
    train(alpha_=p['a'], beta_=p['b'])
