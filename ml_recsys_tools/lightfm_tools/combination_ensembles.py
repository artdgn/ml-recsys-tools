from functools import partial
from itertools import repeat
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import scipy.stats

from ml_recsys_tools.lightfm_tools.similarity_recommenders import SimilarityDFRecommender
from ml_recsys_tools.utils.debug import log_time_and_shape
from ml_recsys_tools.utils.parallelism import batch_generator
from ml_recsys_tools.lightfm_tools.ensembles_base import CombinationEnsembleBase


class CombinedRankEnsemble(CombinationEnsembleBase):

    def __init__(self, recommenders, fill_na_val=None,
                 rank_combination_mode='hmean', *args, **kwargs):
        super().__init__(recommenders=recommenders, *args, **kwargs)
        self.fill_na_val = fill_na_val
        self.rank_combination_mode = rank_combination_mode

    @staticmethod
    def rank_combination_function(mode):
        if mode == 'mean':
            return np.mean
        elif mode == 'max':
            return np.max
        elif mode == 'min':
            return np.min
        elif mode == 'gmean':
            return scipy.stats.gmean
        elif mode == 'hmean':
            return scipy.stats.hmean
        else:
            raise ValueError('Unknown rank_combination_mode: ' + mode)

    @staticmethod
    def calc_dfs_and_combine_scores(calc_funcs, combine_func, fill_val,
                                    groupby_col, item_col, scores_col, multithreaded=False):

        dfs = []
        if multithreaded:
            with ThreadPool(len(calc_funcs)) as pool:
                dfs = pool.map(lambda f: f(), calc_funcs)

        merged_df = None
        rank_cols = []

        for i, func in enumerate(calc_funcs):
            if dfs:
                df = dfs[i]
            else:
                df = func()

            rank_cols.append('rank_' + str(i))
            df[rank_cols[-1]] = df.groupby(groupby_col)[scores_col].rank(ascending=False)
            df.drop(scores_col, axis=1, inplace=True)

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=[groupby_col, item_col], how='outer')

        merged_df.fillna(fill_val, inplace=True)

        # combine ranks
        merged_df[scores_col] = combine_func(1 / merged_df[rank_cols].values, axis=1)

        # drop temp cols
        merged_df.drop(rank_cols, axis=1, inplace=True)

        return merged_df

    def _get_recommendations_flat_unfilt(self, user_ids, n_rec_unfilt, pbar=None, **kwargs):

        calc_funcs = [partial(rec._get_recommendations_flat_unfilt,
                              user_ids=user_ids, n_rec_unfilt=n_rec_unfilt, **kwargs)
                      for rec in self.recommenders]
        return self.calc_dfs_and_combine_scores(
            calc_funcs=calc_funcs,
            combine_func=self.rank_combination_function(self.rank_combination_mode),
            fill_val=self.fill_na_val if self.fill_na_val else (n_rec_unfilt + 1),
            groupby_col=self._user_col,
            item_col=self._item_col,
            scores_col=self._prediction_col)

    def get_similar_items(self, itemids, N=10, n_unfilt=100, results_format='lists', **kwargs):

        calc_funcs = [partial(rec.get_similar_items,
                              itemids=itemids, N=n_unfilt, results_format='flat', **kwargs)
                      for rec in self.recommenders]

        combined_simil_df = self.calc_dfs_and_combine_scores(
            calc_funcs=calc_funcs,
            combine_func=self.rank_combination_function(self.rank_combination_mode),
            fill_val=self.fill_na_val if self.fill_na_val else (n_unfilt + 1),
            groupby_col=self._item_col_simil,
            item_col=self._item_col,
            scores_col=self._prediction_col)

        return combined_simil_df if results_format == 'flat' \
            else self._simil_flat_to_lists(combined_simil_df, n_cutoff=N)


class CombinedSimilRecoEns(SimilarityDFRecommender):

    def __init__(self,
                 recommenders,
                 similarity_func_params=None,
                 n_unfilt=100,
                 numeric_n_bins=30,
                 combination_mode='hmean',
                 use_external_feat_simil=True,
                 use_lfm_simil=True,
                 use_cooc_simil=True,
                 *args, **kwargs):
        self.recommenders = recommenders
        self.similarity_func_params = similarity_func_params
        self.n_unfilt = n_unfilt
        self.combination_mode = combination_mode
        self.numeric_n_bins = numeric_n_bins
        self.use_external_feat_simil = use_external_feat_simil
        self.use_lfm_simil = use_lfm_simil
        self.use_cooc_simil = use_cooc_simil
        super().__init__(*args, **kwargs)

    def set_params(self, **params):
        params = self._pop_set_params(
            params, ['n_unfilt', 'numeric_n_bins',
                     'combination_mode'])
        super().set_params(**params)

    def _get_similarity_func_params(self):
        if self.similarity_func_params is None:
            return repeat({}, len(self.recommenders))
        elif isinstance(self.similarity_func_params, dict):
            return repeat(self.similarity_func_params, len(self.recommenders))
        elif hasattr(self.similarity_func_params, '__len__'):
            return self.similarity_func_params
        else:
            raise ValueError('Unsupported format for similarity functions parameters: %s'
                             % str(self.similarity_func_params))

    @log_time_and_shape
    def fit(self, train_obs, batch_size=10000,
            similarity_queue=None, similarity_queue_cutoff=10, **fit_params):

        itemids = self.recommenders[0].all_training_items()

        for i, items in enumerate(batch_generator(itemids, batch_size)):

            calc_funcs = [
                partial(rec.get_similar_items,
                        itemids=items, N=self.n_unfilt, results_format='flat', **params)
                for rec, params in zip(self.recommenders, self._get_similarity_func_params())]

            simil_df = CombinedRankEnsemble.calc_dfs_and_combine_scores(
                calc_funcs=calc_funcs,
                combine_func=CombinedRankEnsemble.rank_combination_function(self.combination_mode),
                fill_val=self.n_unfilt + 1,
                groupby_col=self._item_col_simil,
                item_col=self._item_col,
                scores_col=self._prediction_col,
                multithreaded=True
            )

            if similarity_queue:
                similarity_queue.put(
                    self._simil_flat_to_lists(
                        simil_df, n_cutoff=similarity_queue_cutoff))

            if i==0:
                super().fit(train_obs, simil_df, **fit_params)
            else:
                super().continue_fit(simil_df)

        if similarity_queue:
            similarity_queue.put('END')

        return self


class CascadeEnsemble(CombinationEnsembleBase):

    def __init__(self, recommenders, *args, **kwargs):
        super().__init__(recommenders, *args, **kwargs)
        assert self.n_recommenders == 2, 'only 2 recommenders supported'

    def _get_recommendations_flat_unfilt(self, user_ids, n_rec_unfilt, pbar=None, **kwargs):
        recos_df = self.recommenders[0]._get_recommendations_flat_unfilt(
            user_ids=user_ids, n_rec_unfilt=n_rec_unfilt, pbar=None, **kwargs)
        return self.recommenders[1].predict_on_df(recos_df)