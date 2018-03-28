from copy import deepcopy

import numpy as np
import pandas as pd
from functools import partial

from lightfm import LightFM
import lightfm.lightfm

from ml_recsys_tools.lightfm_tools.interaction_handlers import RANDOM_STATE
from ml_recsys_tools.utils.automl import early_stopping_runner
from ml_recsys_tools.utils.debug import log_time_and_shape, simple_logger
from ml_recsys_tools.utils.parallelism import map_batches_multiproc, N_CPUS
from ml_recsys_tools.utils.similarity import most_similar, top_N_sorted, top_N_sorted_on_sparse
from ml_recsys_tools.lightfm_tools.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.lightfm_tools.similarity_recommenders import interactions_mat_to_cooccurrence_mat

# monkey patch print function
lightfm.lightfm.print = simple_logger.info


class LightFMRecommender(BaseDFSparseRecommender):
    default_fit_params = {
        'epochs': 100,
        'item_features': None,
        'num_threads': N_CPUS,
        'verbose': True,
    }

    def __init__(self, use_sample_weight=False, external_features_params=None, *args, **kwargs):
        self.use_sample_weight = use_sample_weight
        self.sample_weight = None
        self.cooc_mat = None
        super().__init__(*args, **kwargs)
        if external_features_params is not None:
            self.add_external_features(**external_features_params)

    def _prep_for_fit(self, train_obs, **fit_params):
        # assign all observation data
        self.sparse_mat_builder = train_obs.get_sparse_matrix_helper()
        self.train_df = train_obs.df_obs
        self.user_train_counts = None
        self.train_mat = self.sparse_mat_builder.build_sparse_interaction_matrix(self.train_df)
        self.sample_weight = self.train_mat.tocoo() if self.use_sample_weight else None

        # add external features if specified
        self.fit_params['item_features'] = self.external_features_mat
        if self.external_features_mat is not None:
            simple_logger.info('Fitting using external features mat: %s'
                               % self.external_features_mat.shape)

        # init model and set params
        self.model = LightFM(**self.model_params)
        self._set_fit_params(fit_params)

    @log_time_and_shape
    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.model.fit(self.train_mat, sample_weight=self.sample_weight, **self.fit_params)
        return self

    @log_time_and_shape
    def fit_partial(self, train_obs, epochs=1):
        fit_params = self._dict_update(self.fit_params, {'epochs': epochs})
        if self.model is None:
            self.fit(train_obs, **fit_params)
        else:
            self.model.fit_partial(
                self.train_mat, sample_weight=self.sample_weight, **fit_params)
        return self

    @log_time_and_shape
    def fit_with_early_stop(self, train_obs, valid_ratio=0.04, refit_on_all=False, metric='AUC',
                            epochs_max=200, epochs_step=10, stop_patience=10,
                            plot_convergence=True, decline_threshold=0.05):

        # split validation data
        sqrt_ratio = valid_ratio ** 0.5
        train_obs_internal, valid_obs = train_obs.split_train_test(
            users_ratio=sqrt_ratio, ratio=sqrt_ratio, random_state=RANDOM_STATE)

        self.model_checkpoint = None

        def check_point_func():
            if not refit_on_all:
                self.model_checkpoint = deepcopy(self.model)

        def score_func():
            self.fit_partial(train_obs_internal,
                             epochs=epochs_step)
            lfm_report = self.eval_on_test_by_ranking(
                valid_obs.df_obs, include_train=False)
            cur_score = lfm_report.loc['lfm test', metric]
            return cur_score

        max_epoch = early_stopping_runner(
            score_func=score_func,
            check_point_func=check_point_func,
            epochs_max=epochs_max,
            epochs_step=epochs_step,
            stop_patience=stop_patience,
            decline_threshold=decline_threshold,
            plot_convergence=plot_convergence
        )

        if not refit_on_all:
            simple_logger.info('Loading best model from checkpoint at %d epochs' % max_epoch)
            self.fit_params = self._dict_update(self.fit_params, {'epochs': max_epoch})
            self.model = self.model_checkpoint
            self.model_checkpoint = None
        else:
            # refit on whole data
            simple_logger.info('Refitting on whole train data for %d epochs' % max_epoch)
            self.fit(train_obs, epochs=max_epoch)

        return self

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        if 'epochs' in params:
            self._set_fit_params({'epochs': params.pop('epochs')})
        if 'use_sample_weight' in params:
            self.use_sample_weight = params.pop('use_sample_weight')
        super().set_params(**params)

    def _get_item_representations(self, mode=None):

        n_items = len(self.sparse_mat_builder.iid_encoder.classes_)

        biases, representations = self.model.get_item_representations(self.fit_params['item_features'])

        if mode is None:
            pass  # default mode

        elif mode == 'external_features':
            external_features_mat = self.external_features_mat

            assert external_features_mat is not None, \
                'Must define and add a feature matrix for "external_features" similarity.'

            representations = external_features_mat

        elif (mode == 'no_features') and (self.fit_params['item_features'] is not None):

            simple_logger.info('LightFM recommender: get_similar_items: "no_features" mode '
                               'assumes ID mat was added and is the last part of the feature matrix.')

            assert self.model.item_embeddings.shape[0] > n_items, \
                'Either no ID matrix was added, or no features added'

            representations = self.model.item_embeddings[-n_items:, :]

        else:
            raise ValueError('Uknown representation mode: %s' % mode)

        return biases, representations

    @log_time_and_shape
    def get_similar_items(self, itemids, N=10, remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists', pbar=None):
        """
        uses learned embeddings to get N most similar items

        :param itemids: vector of item IDs
        :param N: number of most similar items to retrieve
        :param remove_self: whether to remove the the query items from the lists (similarity to self should be maximal)
        :param embeddings_mode: the item representations to use for calculation:
             None (default) - means full representations
             'external_features' - calculation based only external features (assumes those exist)
             'no_features' - calculation based only on internal features (assumed identity mat was part of the features)
        :param simil_mode: mode of similairyt calculation:
            'cosine' (default) - cosine similarity bewtween representations (normalized dot product with no biases)
            'dot' - unnormalized dot product with addition of biases
            'euclidean' - inverse of euclidean distance
            'cooccurance' - no usage of learned features - just cooccurence of items matrix
                (number of 2nd degree connections in user-item graph)
        :param results_format:
            'flat' for dataframe of triplets (source_item, similar_item, similarity)
            'lists' for dataframe of lists (source_item, list of similar items, list of similarity scores)
        :param pbar: name of tqdm progress bar (None means no tqdm)

        :return: a matrix of most similar IDs [n_ids, N], a matrix of score of those similarities [n_ids, N]
        """

        if simil_mode in ['cosine', 'dot', 'euclidean']:
            biases, representations = self._get_item_representations(mode=embeddings_mode)

            best_ids, best_scores = most_similar(
                ids=itemids,
                source_encoder=self.sparse_mat_builder.iid_encoder,
                source_mat=representations,
                source_biases=biases,
                n=N,
                remove_self=remove_self,
                simil_mode=simil_mode,
                pbar=pbar
            )

        elif simil_mode == 'cooccurrence':
            if self.cooc_mat is None or \
                    self.cooc_mat.shape[1] != self.train_mat.shape[1]:
                self.cooc_mat = interactions_mat_to_cooccurrence_mat(self.train_mat)

            best_ids, best_scores = top_N_sorted_on_sparse(
                ids=itemids,
                encoder=self.sparse_mat_builder.iid_encoder,
                sparse_mat=self.cooc_mat,
                n_top=N
            )

        else:
            raise ValueError('Unknown similarity mode: %s' % simil_mode)

        simil_df = self._format_results_df(
            itemids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_' + results_format)

        return simil_df

    @log_time_and_shape
    def get_similar_users(self, userids, N=10, remove_self=True, simil_mode='cosine', pbar=None):
        """
        same as get_similar_items but for users
        """
        user_biases, user_representations = self.model.get_user_representations()
        best_ids, best_scores = most_similar(
            ids=userids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            source_mat=user_representations,
            source_biases=user_biases,
            n=N,
            remove_self=remove_self,
            simil_mode=simil_mode,
            pbar=pbar
        )

        simil_df = self._format_results_df(
            userids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_lists'). \
            rename({self._item_col_simil: self._user_col})
        # this is UGLY, if this function is ever used, fix this please (the renaming shortcut)

        return simil_df

    def _get_recommendations_flat_unfilt(
            self, user_ids, n_rec_unfilt, pbar=None, item_features_mode=None, use_biases=True):

        user_biases, user_representations = self.model.get_user_representations()
        item_biases, item_representations = self._get_item_representations(mode=item_features_mode)

        if not use_biases:
            user_biases, item_biases = None, None

        best_ids, best_scores = most_similar(
            ids=user_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            target_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=user_representations,
            target_mat=item_representations,
            source_biases=user_biases,
            target_biases=item_biases,
            n=n_rec_unfilt,
            remove_self=False,
            simil_mode='dot',
            pbar=pbar
        )

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    @log_time_and_shape
    def predict_on_df(self, df):
        mat_builder = self.get_prediction_mat_builder_adapter(self.sparse_mat_builder)
        df = mat_builder.add_encoded_cols(df)
        df[self._prediction_col] = self.model.predict(
            df[mat_builder.uid_col].values,
            df[mat_builder.iid_col].values,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads'])
        df.drop([mat_builder.uid_col, mat_builder.iid_col], axis=1, inplace=True)
        return df

    @log_time_and_shape
    def eval_on_test_by_ranking_exact_and_slow(self, test_dfs, test_names=('',), prefix='lfm ', include_train=True):

        @log_time_and_shape
        def _get_training_ranks():
            ranks_mat = self.model.predict_rank(
                self.train_mat,
                item_features=self.fit_params['item_features'],
                num_threads=self.fit_params['num_threads'])
            return ranks_mat, self.train_mat

        @log_time_and_shape
        def _get_test_ranks(test_df):
            test_sparse = self.sparse_mat_builder.build_sparse_interaction_matrix(test_df)
            ranks_mat = self.model.predict_rank(
                test_sparse, train_interactions=self.train_mat,
                item_features=self.fit_params['item_features'],
                num_threads=self.fit_params['num_threads'])
            return ranks_mat, test_sparse

        return self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_tanks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train)

    @log_time_and_shape
    def get_recommendations_exact_and_slow(
            self, user_ids, n_rec=10, exclude_training=True, chunksize=200, results_format='lists'):

        calc_func = partial(
            self._get_recommendations_exact_and_slow,
            n_rec=n_rec,
            exclude_training=exclude_training,
            results_format=results_format)

        chunksize = int(35000 * chunksize / self.sparse_mat_builder.n_cols)

        ret = map_batches_multiproc(
            calc_func, user_ids, chunksize=chunksize, pbar='get_recommendations_exact_and_slow')
        return pd.concat(ret, axis=0)

    def _predict_for_users_dense(self, user_ids, exclude_training):

        mat_builder = self.sparse_mat_builder
        n_items = mat_builder.n_cols

        user_inds = mat_builder.uid_encoder.transform(user_ids)

        n_users = len(user_inds)
        user_inds_mat = user_inds.repeat(n_items)
        item_inds_mat = np.tile(np.arange(n_items), n_users)

        full_pred_mat = self.model.predict(
            user_inds_mat,
            item_inds_mat,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads']). \
            reshape((n_users, n_items))

        train_mat = self.train_mat.tocsr()

        if exclude_training:
            train_mat.sort_indices()
            for pred_ind, user_ind in enumerate(user_inds):
                train_inds = train_mat.indices[
                             train_mat.indptr[user_ind]: train_mat.indptr[user_ind + 1]]
                full_pred_mat[pred_ind, train_inds] = -np.inf

        return full_pred_mat

    def _get_recommendations_exact_and_slow(self, user_ids, n_rec=10, exclude_training=True,
                                            results_format='lists'):

        full_pred_mat = self._predict_for_users_dense(user_ids, exclude_training=exclude_training)

        top_scores, top_inds = top_N_sorted(full_pred_mat, n=n_rec)

        item_ids = self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds)

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=item_ids,
            scores_mat=top_scores, results_format='recommendations_' + results_format)
