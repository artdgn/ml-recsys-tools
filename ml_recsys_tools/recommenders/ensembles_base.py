import warnings
from abc import abstractmethod, ABC
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool

import numpy as np
import pandas as pd

from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.parallelism import N_CPUS


class SubdivisionEnsembleBase(BaseDFSparseRecommender, ABC):

    def __init__(self,
                 n_models=1,
                 max_concurrent=4,
                 normalize_predictions=True,
                 concurrency_backend='threads', **kwargs):
        self.n_models = n_models
        self.max_concurrent = max_concurrent
        self.concurrency_backend = concurrency_backend
        self.normalize_predictions = normalize_predictions
        super().__init__(**kwargs)
        self.sub_class_type = None
        self._init_sub_models()

    def get_workers_pool(self, concurrency_backend=None):
        if concurrency_backend is None:
            concurrency_backend = self.concurrency_backend
        if 'thread' in concurrency_backend:
            return ThreadPool(self.n_concurrent())
        elif 'proc' in concurrency_backend:
            return Pool(self.n_concurrent(), maxtasksperchild=3)

    def _init_sub_models(self, **params):
        self.sub_models = [self.sub_class_type(**params.copy())
                           for _ in range(self.n_models)]

    def n_concurrent(self):
        return min(self.n_models, self.max_concurrent, N_CPUS)

    def _broadcast(self, var):
        if isinstance(var, list) and len(var) == self.n_models:
            return var
        else:
            return [var] * self.n_models

    def set_params(self, **params):
        params = self._pop_set_params(
            params, ['n_models'])
        # set on self
        super().set_params(**params.copy())
        # init sub models to make sure they're the right object already
        self._init_sub_models()
        # set for each sub_model
        for model in self.sub_models:
            model.set_params(**params.copy())

    @abstractmethod
    def _generate_sub_model_train_data(self, train_obs):
        pass

    @abstractmethod
    def _fit_sub_model(self, args):
        pass

    def fit(self, train_obs, **fit_params):
        self._set_data(train_obs)

        sub_model_train_data_generator = self._generate_sub_model_train_data(train_obs)

        with self.get_workers_pool() as pool:
            self.sub_models = list(
                pool.imap(self._fit_sub_model,
                          zip(range(self.n_models),
                              sub_model_train_data_generator,
                              repeat(fit_params, self.n_models))))
        return self

    def _get_recommendations_flat(self, user_ids, item_ids, n_rec=100, **kwargs):

        def _calc_recos_sub_model(i_model):
            all_users = np.array(self.sub_models[i_model].all_users)
            users = all_users[np.isin(all_users, user_ids)]
            reco_df = pd.DataFrame()
            if len(users):
                reco_df = self.sub_models[i_model].get_recommendations(
                    user_ids=users, item_ids=item_ids,
                    n_rec=n_rec, results_format='flat', **kwargs)
                if self.normalize_predictions:
                    reco_df[self._prediction_col].values /= reco_df[self._prediction_col].max()
            return reco_df

        with self.get_workers_pool('threads') as pool:
            reco_dfs = pool.map(_calc_recos_sub_model, np.arange(len(self.sub_models)))

        recos_flat = pd.concat(reco_dfs, axis=0). \
            sort_values(self._prediction_col, ascending=False). \
            drop_duplicates(subset=[self._user_col, self._item_col], keep='first')

        return recos_flat

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists', **kwargs):

        def _calc_simils_sub_model(i_model):
            all_items = np.array(self.sub_models[i_model].all_items)
            items = all_items[np.isin(all_items, item_ids)]
            simil_df = pd.DataFrame()
            if len(items):
                simil_df = self.sub_models[i_model].get_similar_items(
                    item_ids=items, target_item_ids=target_item_ids,
                    n_simil=n_simil, remove_self=remove_self,
                    embeddings_mode=embeddings_mode, simil_mode=simil_mode,
                    results_format='flat', pbar=None)
                if self.normalize_predictions:
                    simil_df[self._prediction_col].values /= simil_df[self._prediction_col].max()
            return simil_df

        with self.get_workers_pool('threads') as pool:
            simil_dfs = pool.map(_calc_simils_sub_model, np.arange(len(self.sub_models)))

        simil_all = pd.concat(simil_dfs, axis=0). \
            sort_values(self._prediction_col, ascending=False). \
            drop_duplicates(subset=[self._item_col_simil, self._item_col], keep='first')

        return simil_all if results_format == 'flat' \
            else self._simil_flat_to_lists(simil_all, n_cutoff=n_simil)

    # def sub_model_evaluations(self, test_dfs, test_names, include_train=True):
    #     stats = []
    #     reports = []
    #     for m in self.sub_models:
    #         users = m.train_df[self.train_obs.uid_col].unique()
    #         items = m.train_df[self.train_obs.iid_col].unique()
    #         sub_test_dfs = [df[df[self.train_obs.uid_col].isin(users) &
    #                            df[self.train_obs.iid_col].isin(items)] for df in test_dfs]
    #         lfm_report = m.eval_on_test_by_ranking(
    #             include_train=include_train,
    #             test_dfs=sub_test_dfs,
    #             prefix='lfm sub model',
    #             test_names=test_names
    #         )
    #         stats.append('train: %d, test: %s' %
    #                      (len(m.train_df), [len(df) for df in sub_test_dfs]))
    #         reports.append(lfm_report)
    #     return stats, reports


class CombinationEnsembleBase(BaseDFSparseRecommender):

    def __init__(self, recommenders, **kwargs):
        self.recommenders = recommenders
        self.n_recommenders = len(self.recommenders)
        super().__init__(**kwargs)
        self._reuse_data(self.recommenders[0])

    def fit(self, *args, **kwargs):
        warnings.warn('Fit is not supported, recommenders should already be fitted.')
