import os
from abc import ABC, abstractmethod
from copy import deepcopy
from time import sleep
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pickle

import time

from ml_recsys_tools.data_handlers.interactions_with_features import ExternalFeaturesDF
from ml_recsys_tools.utils.logger import simple_logger as logger
from ml_recsys_tools.utils.automl import BayesSearchHoldOut, SearchSpaceGuess
from ml_recsys_tools.evaluation.ranks_scoring import mean_scores_report_on_ranks
from ml_recsys_tools.data_handlers.interaction_handlers_base import InteractionMatrixBuilder, RANDOM_STATE, \
    ObservationsDF
from ml_recsys_tools.utils.instrumentation import LogCallsTimeAndOutput


class BaseDFRecommender(ABC, LogCallsTimeAndOutput):
    default_model_params = {}
    default_fit_params = {}

    def __init__(self, user_col='userid', item_col='itemid',
                 rating_col='rating', prediction_col='prediction',
                 model_params=None, fit_params=None, verbose=True):
        super().__init__(verbose)
        self._user_col = user_col
        self._item_col = item_col
        self._item_col_simil = item_col + '_source'
        self._rating_col = rating_col
        self._prediction_col = prediction_col
        self.model_params = self.default_model_params.copy()
        self.fit_params = self.default_fit_params.copy()
        self._set_model_params(model_params)
        self._set_fit_params(fit_params)
        self.train_df = None
        self.model = None

    @classmethod
    def guess_search_space(cls):
        return SearchSpaceGuess(cls)

    @staticmethod
    def _dict_update(d, u):
        d = d.copy()
        if u: d.update(u)
        return d

    @staticmethod
    def _pop_set_dict(d, params, pop_params):
        for p in pop_params:
            if p in params:
                d[p] = params.pop(p)
        return params

    def _pop_set_params(self, params, pop_params):
        return self._pop_set_dict(self.__dict__, params, pop_params)

    def _set_model_params(self, params):
        self.model_params = self._dict_update(self.model_params, params)

    def _set_fit_params(self, params):
        self.fit_params = self._dict_update(self.fit_params, params)

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        self._set_model_params(params)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_recommendations(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval_on_test_by_ranking(self, *args, **kwargs):
        pass

    def _recos_lists_to_flat(self, recos_lists_df):

        # n_users = len(recos_lists_df)
        n_recos = len(recos_lists_df[self._item_col].iloc[0])

        recos_df_flat = pd.DataFrame({
            self._user_col: recos_lists_df[self._user_col].values.repeat(n_recos),
            self._item_col: np.concatenate(recos_lists_df[self._item_col].values),
            self._prediction_col: np.concatenate(recos_lists_df[self._prediction_col].values),
        })

        return recos_df_flat

    @staticmethod
    def _flat_df_to_lists(df, sort_col, group_col, n_cutoff, target_columns):
        order = [group_col] + target_columns
        return df[order]. \
            sort_values(sort_col, ascending=False). \
            groupby(group_col). \
            aggregate(lambda x: list(x)[:n_cutoff]). \
            reset_index()

    def _recos_flat_to_lists(self, df, n_cutoff=None):
        return self._flat_df_to_lists(
            df,
            n_cutoff=n_cutoff,
            sort_col=self._prediction_col,
            group_col=self._user_col,
            target_columns=[self._item_col, self._prediction_col])

    def _simil_flat_to_lists(self, df, n_cutoff=None):
        return self._flat_df_to_lists(
            df,
            n_cutoff=n_cutoff,
            sort_col=self._prediction_col,
            group_col=self._item_col_simil,
            target_columns=[self._item_col, self._prediction_col])

    def _format_results_df(self, source_vec, target_ids_mat, scores_mat, results_format):
        if 'recommendations' in results_format:
            source_col = self._user_col
            target_col = self._item_col
            scores_col = self._prediction_col
        elif 'similarities' in results_format:
            source_col = self._item_col_simil
            target_col = self._item_col
            scores_col = self._prediction_col
        else:
            raise NotImplementedError('results_format: ' + results_format)

        order = [source_col, target_col, scores_col]

        if 'lists' in results_format:
            return pd.DataFrame({
                source_col: source_vec,
                target_col: list(target_ids_mat),
                scores_col: list(scores_mat)
            })[order]
        elif 'flat' in results_format:
            n_rec = target_ids_mat.shape[1]
            return pd.DataFrame({
                source_col: np.array(source_vec).repeat(n_rec),
                target_col: np.concatenate(list(target_ids_mat)),
                scores_col: np.concatenate(list(scores_mat)),
            })[order]

        else:
            raise NotImplementedError('results_format: ' + results_format)

    def hyper_param_search_on_split_data(
            self, train_data, validation_data, hp_space, n_iters=100, random_search=0.9, **kwargs):

        bo = RecoBayesSearchHoldOut(search_space=hp_space,
                                    pipeline=self,
                                    loss=None,
                                    random_ratio=random_search,
                                    **kwargs)

        res_bo, best_params, best_model = \
            bo.optimize(data_dict={'training': train_data, 'validation': validation_data},
                        n_calls=n_iters)

        bo_report = bo.best_results_summary(res_bo, percentile=0)

        return SimpleNamespace(**{
            'optimizer': bo,
            'report': bo_report,
            'result': res_bo,
            'best_params': best_params,
            'best_model': best_model})

    def pickle_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def hyper_param_search(self, train_obs, hp_space, n_iters=100, valid_ratio=0.04, random_search=0.9, **kwargs):

        sqrt_ratio = valid_ratio ** 0.5
        train_obs_bo, valid_obs_bo = train_obs.split_train_test(
            users_ratio=sqrt_ratio, ratio=sqrt_ratio, random_state=RANDOM_STATE)

        return self.hyper_param_search_on_split_data(
            train_data=train_obs_bo, validation_data=valid_obs_bo.df_obs,
            hp_space=hp_space, n_iters=n_iters, random_search=random_search, **kwargs)


class BaseDFSparseRecommender(BaseDFRecommender):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sparse_mat_builder = None
        self.train_mat = None
        self.user_train_counts = None
        self.external_features_mat = None

    def all_users(self):
        return self.train_df[self.sparse_mat_builder.uid_source_col].\
            unique().astype(str)

    def all_items(self):
        return self.train_df[self.sparse_mat_builder.iid_source_col].\
            unique().astype(str)

    def remove_unseen_users(self, user_ids, message_prefix=''):
        return self._filter_array(
            user_ids,
            filter_array=self.all_users(),
            message_prefix=message_prefix,
            message_suffix='useres that were not in training set.')

    def remove_unseen_items(self, item_ids, message_prefix=''):
        return self._filter_array(
            item_ids,
            filter_array=self.all_items(),
            message_prefix=message_prefix,
            message_suffix='items that were not in training set.')

    @staticmethod
    def _filter_array(array, filter_array, message_prefix='', message_suffix=''):
        array = np.array(array).astype(str)
        relevance_mask = np.isin(array, np.array(filter_array).astype(str))
        n_discard = np.sum(~relevance_mask)
        if n_discard > 0:
            logger.info(
                '%s Discarding %d (out of %d) %s' %
                (message_prefix, int(n_discard), len(array), message_suffix))
        return array[relevance_mask]

    def _remove_training_from_df(self, flat_df):
        flat_df = pd.merge(
            flat_df, self.train_df,
            left_on=[self._user_col, self._item_col],
            right_on=[self._user_col, self.sparse_mat_builder.iid_source_col],
            how='left')
        # keep only data that was present in left (recommendations) but no in right (training)
        flat_df = flat_df[flat_df[self._rating_col].isnull()] \
            [[self._user_col, self._item_col, self._prediction_col]]
        return flat_df

    def _remove_self_similarities(self, flat_df, col1, col2):
        return flat_df[flat_df[col1].values != flat_df[col2].values].copy()

    @staticmethod
    def _eval_on_test_by_ranking_LFM(train_ranks_func, test_tanks_func,
                                     test_dfs, test_names=('',), prefix='', include_train=True):
        """
        this is just to avoid the same flow twice (or more)
        :param train_ranks_func: function that return the ranks and sparse mat of training set
        :param test_tanks_func: function that return the ranks and sparse mat of a test set
        :param test_dfs: test dataframes
        :param test_names: test dataframes names
        :param prefix: prefix for this report
        :param include_train: whether to evaluate training or not
        :return: a report dataframe
        """

        # test
        if isinstance(test_dfs, pd.DataFrame):
            test_dfs = [test_dfs]

        report_dfs = []

        # train
        if include_train:
            ranks_mat, sp_mat = train_ranks_func()
            report_dfs.append(mean_scores_report_on_ranks(
                ranks_list=[ranks_mat], datasets=[sp_mat], dataset_names=[prefix + 'train']))

        for test_df, test_name in zip(test_dfs, test_names):
            ranks_mat, sp_mat = test_tanks_func(test_df)
            report_dfs.append(mean_scores_report_on_ranks(
                ranks_list=[ranks_mat], datasets=[sp_mat], dataset_names=[prefix + test_name + 'test']))

        report_df = pd.concat(report_dfs)
        return report_df

    def get_prediction_mat_builder_adapter(self, mat_builder: InteractionMatrixBuilder):
        mat_builder = deepcopy(mat_builder)
        mat_builder.uid_source_col = self._user_col
        mat_builder.iid_source_col = self._item_col
        mat_builder.rating_source_col = self._prediction_col
        return mat_builder

    def _separate_heavy_users(self, user_ids, threshold=100):
        # calculate the user training counts if not calculated
        if self.user_train_counts is None:
            self.user_train_counts = self.train_df.userid.value_counts()

        user_counts = self.user_train_counts[self.user_train_counts.index.isin(user_ids)]

        heavy_user_counts = user_counts[user_counts >= threshold]
        heavy_users_max = heavy_user_counts.max()
        heavy_users = heavy_user_counts.index.tolist()
        normal_users = user_counts[user_counts < threshold].index.tolist()

        return heavy_users, normal_users, heavy_users_max

    @abstractmethod
    def _get_recommendations_flat_unfilt(self, user_ids, item_ids, n_rec_unfilt,
                                         exclude_training=True, pbar=None, **kwargs):
        pass

    def get_recommendations(
            self, user_ids=None, item_ids=None, n_rec=10, n_rec_unfilt=100,
            exclude_training=True, pbar=None,
            results_format='lists'):

        if user_ids is not None:
            user_ids = self.remove_unseen_users(user_ids, message_prefix='get_recommendations: ')
        else:
            user_ids = self.all_users()

        if item_ids is not None:
            item_ids = self.remove_unseen_items(item_ids, message_prefix='get_recommendations: ')
        else:
            item_ids = self.all_items()

        # treat heavy users differently
        heavy_users, normal_users, heavy_users_max = \
            self._separate_heavy_users(user_ids=user_ids, threshold=50)
        reco_dfs = []
        for user_group, n_unfilt in zip([heavy_users, normal_users],
                                        [n_rec_unfilt + heavy_users_max, n_rec_unfilt]):
            if len(user_group):
                reco_dfs.append(
                    self._get_recommendations_flat_unfilt(
                        user_ids=user_group,
                        item_ids=item_ids,
                        n_rec_unfilt=n_unfilt,
                        pbar=pbar))

        recos_flat = pd.concat(reco_dfs, axis=0)

        del reco_dfs

        if exclude_training:
            recos_flat = self._remove_training_from_df(recos_flat)

        if results_format == 'flat':
            return recos_flat
        else:
            return self._recos_flat_to_lists(recos_flat, n_cutoff=n_rec)

    def _check_item_ids_args(self, item_ids, target_item_ids):
        if item_ids is None:
            item_ids = self.all_items()
        else:
            item_ids = self.remove_unseen_items(item_ids)
        if target_item_ids is not None:
            target_item_ids = self.remove_unseen_items(target_item_ids)
        return item_ids, target_item_ids

    def eval_on_test_by_ranking(self, test_dfs, test_names=('',), prefix='rec ',
                                include_train=True, items_filter=None,
                                n_rec=10, n_rec_unfilt=200):
        @self.logging_decorator
        def relevant_users():
            # get only those users that are present in the evaluation / training dataframes
            all_test_users = []
            [all_test_users.extend(df[self._user_col].unique().tolist()) for df in test_dfs]
            all_test_users = np.array(all_test_users)
            return all_test_users

        if isinstance(test_dfs, pd.DataFrame):
            test_dfs = [test_dfs]
        elif isinstance(test_dfs, ObservationsDF):
            test_dfs = [test_dfs.df_obs]

        mat_builder = self.sparse_mat_builder
        pred_mat_builder = self.get_prediction_mat_builder_adapter(mat_builder)

        users = relevant_users()

        recos_flat_unfilt = self.get_recommendations(
            user_ids=users,
            item_ids=items_filter,
            n_rec_unfilt=min(n_rec_unfilt, mat_builder.n_cols),
            exclude_training=(not include_train),
            results_format='flat')

        if include_train:
            ranks_all_no_train = pred_mat_builder.predictions_df_to_sparse_ranks(
                self._remove_training_from_df(recos_flat_unfilt))
        else:
            ranks_all_no_train = pred_mat_builder.predictions_df_to_sparse_ranks(
                recos_flat_unfilt)

        @self.logging_decorator
        def _get_training_ranks():
            train_df = self.train_df[self.train_df[self._user_col].isin(users)].copy()
            sp_train = self.sparse_mat_builder. \
                build_sparse_interaction_matrix(train_df).tocsr()
            sp_train_ranks = pred_mat_builder. \
                filter_all_ranks_by_sparse_selection(
                sp_train, pred_mat_builder.predictions_df_to_sparse_ranks(recos_flat_unfilt))
            return sp_train_ranks, sp_train

        @self.logging_decorator
        def _get_test_ranks(test_df):
            sp_test = self.sparse_mat_builder. \
                build_sparse_interaction_matrix(test_df).tocsr()
            sp_test_ranks = pred_mat_builder. \
                filter_all_ranks_by_sparse_selection(sp_test, ranks_all_no_train)
            return sp_test_ranks, sp_test

        report_df = self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_tanks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train)

        return report_df


class RecoBayesSearchHoldOut(BayesSearchHoldOut, LogCallsTimeAndOutput):

    def __init__(self, metric='AUC', interrupt_message_file=None, verbose=True, **kwargs):
        self.metric = metric
        self.interrupt_message_file = interrupt_message_file
        super().__init__(verbose=verbose, **kwargs)
        self.all_metrics = pd.DataFrame()

    def _check_interrupt(self):
        if self.interrupt_message_file is not None \
                and os.path.exists(self.interrupt_message_file):

            with open(self.interrupt_message_file) as f:
                message = f.readline()

            if 'stop' in message:
                raise InterruptedError('interrupted by "stop" message in %s'
                                       % self.interrupt_message_file)
            elif 'pause' in message:
                logger.warn('Paused by "pause" message in %s'
                            % self.interrupt_message_file)
                while 'pause' in message:
                    sleep(1)
                    with open(self.interrupt_message_file) as f:
                        message = f.readline()
                self._check_interrupt()

            elif 'update' in message:
                logger.warn('Updating HP space due to "update" message in %s'
                            % self.interrupt_message_file)
                raise NotImplementedError('not yet implemented')

    def _record_all_metrics(self, report_df, values, time_taken, target_loss):
        # records the time and the other metrics
        params_dict = self.values_to_dict(values)
        report_df['target_loss'] = target_loss
        report_df['time_taken'] = time_taken
        report_df = report_df.assign(**params_dict)
        self.all_metrics = self.all_metrics.append(report_df)

    def best_results_summary(self, res_bo, percentile=95):
        return self.all_metrics. \
            reset_index(). \
            drop('index', axis=1). \
            sort_values('target_loss')

    def objective_func(self, values):
        try:
            self._check_interrupt()
            pipeline = self.init_pipeline(values)
            start = time.time()
            pipeline.fit(self.data_dict['training'])
            report_df = pipeline.eval_on_test_by_ranking(
                self.data_dict['validation'], include_train=False, prefix='')
            loss = 1 - report_df.loc['test', self.metric]
            self._record_all_metrics(
                report_df=report_df, values=values,
                time_taken=time.time() - start, target_loss=loss)
            return loss

        except Exception as e:
            logger.exception(e)
            logger.error(values)
            # raise e
            return 1.0
