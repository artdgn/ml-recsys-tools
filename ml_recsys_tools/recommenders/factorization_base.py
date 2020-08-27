from abc import abstractmethod

import pandas as pd
import numpy as np
import scipy.sparse as sp

from ml_recsys_tools.recommenders.recommender_base import BasePredictorRecommender
from ml_recsys_tools.utils.similarity import most_similar


class BaseFactorizationRecommender(BasePredictorRecommender):

    @abstractmethod
    def _get_item_factors(self, mode=None):
        return  np.array([0]), np.array([0])

    @abstractmethod
    def _get_user_factors(self, mode=None):
        return np.array([0]), np.array([0])

    @abstractmethod
    def _prep_for_fit(self, train_obs, **fit_params):
        pass

    @abstractmethod
    def fit_partial(self, train_obs, **fit_params):
        pass

    @abstractmethod
    def _set_epochs(self, epochs):
        pass

    def _factors_to_dataframe(self, factor_func, include_biases=False):
        b, f = factor_func()
        return pd.DataFrame(
            index=np.arange(f.shape[0]),
            data=np.concatenate([b, f], axis=1) if include_biases else f)

    def user_factors_dataframe(self, include_biases=False):
        return self._factors_to_dataframe(self._get_user_factors, include_biases=include_biases)

    def item_factors_dataframe(self, include_biases=False):
        return self._factors_to_dataframe(self._get_item_factors, include_biases=include_biases)

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists'):
        """
        uses learned embeddings to get N most similar items

        :param item_ids: vector of item IDs
        :param n_simil: number of most similar items to retrieve
        :param remove_self: whether to remove the the query items from the lists (similarity to self should be maximal)
        :param embeddings_mode: the item representations to use for calculation:
             None (default) - means full representations
             'external_features' - calculation based only external features (assumes those exist)
             'no_features' - calculation based only on internal features (assumed identity mat was part of the features)
        :param simil_mode: mode of similairyt calculation:
            'cosine' (default) - cosine similarity bewtween representations (normalized dot product with no biases)
            'dot' - unnormalized dot product with addition of biases
            'euclidean' - inverse of euclidean distance
        :param results_format:
            'flat' for dataframe of triplets (source_item, similar_item, similarity)
            'lists' for dataframe of lists (source_item, list of similar items, list of similarity scores)

        :return: a matrix of most similar IDs [n_ids, N], a matrix of score of those similarities [n_ids, N]
        """

        item_ids, target_item_ids = self._check_item_ids_args(item_ids, target_item_ids)

        biases, representations = self._get_item_factors(mode=embeddings_mode)

        best_ids, best_scores = most_similar(
            source_ids=item_ids,
            target_ids=target_item_ids,
            source_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=representations,
            source_biases=biases,
            n=n_simil+1 if remove_self else n_simil,
            simil_mode=simil_mode,
        )

        simil_df = self._format_results_df(
            item_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_flat')

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._item_col_simil, col2=self._item_col)

        if 'lists' in results_format:
            simil_df = self._simil_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df

    def get_similar_users(self, user_ids=None, target_user_ids=None, n_simil=10, remove_self=True,
                          simil_mode='cosine'):
        """
        same as get_similar_items but for users
        """
        user_ids, target_user_ids = self._check_user_ids_args(user_ids, target_user_ids)

        user_biases, user_representations = self._get_user_factors()

        best_ids, best_scores = most_similar(
            source_ids=user_ids,
            target_ids=target_user_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            source_mat=user_representations,
            source_biases=user_biases,
            n=n_simil,
            simil_mode=simil_mode,
        )

        simil_df = self._format_results_df(
            user_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_flat'). \
            rename({self._item_col_simil: self._user_col}, axis=1)
        # this is UGLY, if this function is ever useful, fix this please (the renaming shortcut)

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._user_col, col2=self._item_col)

        simil_df = self._recos_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df

    def _get_recommendations_flat(
            self, user_ids, n_rec, item_ids=None, exclusions=True,
            item_features_mode=None, use_biases=True):

        user_biases, user_representations = self._get_user_factors()
        item_biases, item_representations = self._get_item_factors(mode=item_features_mode)

        if not use_biases:
            user_biases, item_biases = None, None

        best_ids, best_scores = most_similar(
            source_ids=user_ids,
            target_ids=item_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            target_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=user_representations,
            target_mat=item_representations,
            source_biases=user_biases,
            target_biases=item_biases,
            exclude_mat_sp=self.exclude_mat if exclusions else None,
            n=n_rec,
            simil_mode='dot',
        )

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    def _predict_for_users_dense_direct(self, user_ids, item_ids=None, exclusions=True):
        """
        method for calculating prediction for a grid of users and items
        directly from the calculated factors. this method is faster for smaller inputs
        an can be further sped up by employing batched multiprocessing (as
        used in similarity / recommendation calculations)

        :param user_ids: users ids
        :param item_ids: item ids, when None - all known items are used
        :param exclusions: when True sets prediction on excluded examples to -np.inf
        :return: a matrix of predictions (n_users, n_items)
        """
        if item_ids is None:
            item_ids = self.sparse_mat_builder.iid_encoder.classes_

        user_inds = self.user_inds(user_ids)
        item_inds = self.item_inds(item_ids)

        user_biases, user_factors = self._get_user_factors()
        item_biases, item_factors = self._get_item_factors()

        scores = np.dot(user_factors[user_inds, :], item_factors[item_inds, :].T)

        if user_biases is not None:
            scores = (scores.T + user_biases[user_inds]).T

        if item_biases is not None:
            scores += item_biases[item_inds]

        if sp.issparse(scores):
            scores = scores.toarray()
        else:
            scores = np.array(scores)

        full_pred_mat = scores

        if exclusions:
            exclude_mat_sp_coo = self.exclude_mat[user_inds, :][:, item_inds].tocoo()
            full_pred_mat[exclude_mat_sp_coo.row, exclude_mat_sp_coo.col] = -np.inf

        return full_pred_mat

    def _predict_for_items_dense_direct(self, items_ids_source, item_ids_target=None):
        """ dense similarity predictions from items to items """
        item_inds_s = self.item_inds(items_ids_source)
        item_inds_t = self.item_inds(item_ids_target)

        item_biases, item_factors = self._get_item_factors()

        scores = np.dot(item_factors[item_inds_s, :], item_factors[item_inds_t, :].T)

        if item_biases is not None:
            scores = (scores.T + item_biases[item_inds_s]).T
            scores += item_biases[item_inds_t]

        if sp.issparse(scores):
            scores = scores.toarray()
        else:
            scores = np.array(scores)
        return scores



