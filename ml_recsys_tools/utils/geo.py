import pprint
import io

import numpy as np
import os
import gmaps
import ipywidgets.embed
import colorsys

from ml_recsys_tools.data_handlers.interactions_with_features import ItemsHandler, ItemsGeoMapper
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender


class ItemsGeoMap:

    def __init__(self, df_items=None, lat_col='lat', long_col='long'):
        self.df_items = df_items
        self.lat_col = lat_col
        self.long_col = long_col
        self.fig = None
        gmaps.configure(api_key=os.environ['GOOGLE_MAPS_API_KEY'])

    @property
    def loc_cols(self):
        return [self.lat_col, self.long_col]

    def _ceter_location(self):
        return self.df_items[self.loc_cols].apply(np.mean).tolist()

    def _zoom_heuristic(self):

        # https://stackoverflow.com/questions/6048975/google-maps-v3-how-to-calculate-the-zoom-level-for-a-given-bounds
        def gm_heuristic(min_deg, max_deg):
            if max_deg - min_deg > 0:
                return int(np.log(1000 * 360 / (max_deg - min_deg) / 256) / np.log(2))
            else:
                return 14

        ranges = self.df_items[self.loc_cols].apply([np.min, np.max]).values

        zoom_level_lat = gm_heuristic(ranges[0][0], ranges[1][0])

        zoom_level_long = gm_heuristic(ranges[0][1], ranges[1][1])

        return min(zoom_level_lat, zoom_level_long)

    def reset_map(self):
        self.fig = None

    def _check_get_view_fig(self):
        if self.fig is None:
            self.fig = gmaps.figure()
            self.fig.widgets.clear()  # clear any history
            self.fig = gmaps.figure(
                center=self._ceter_location(),
                zoom_level=self._zoom_heuristic())

    def add_heatmap(self,
                    df_items=None,
                    color=(0, 250, 50),
                    opacity=0.6,
                    sensitivity=5,
                    spread=30, ):

        self._check_get_view_fig()

        if df_items is None:
            df_items = self.df_items

        self.fig.add_layer(
            gmaps.heatmap_layer(
                df_items[self.loc_cols].values,
                opacity=opacity,
                max_intensity=sensitivity,
                point_radius=spread,
                dissipating=True,
                gradient=[list(color) + [0],
                          list(color) + [1]]))
        return self

    def add_markers(self,
                    df_items=None,
                    max_markers=1000,
                    color='red',
                    size=2,
                    opacity=1.0,
                    fill=True
                    ):

        self._check_get_view_fig()

        if df_items is None:
            df_items = self.df_items

        marker_locs, marker_info = self._markers_with_info(df_items, max_markers=max_markers)

        self.fig.add_layer(gmaps.symbol_layer(
            marker_locs[self.loc_cols].values,
            fill_color=color,
            stroke_color=color,
            fill_opacity=opacity if fill else 0,
            stroke_opacity=opacity,
            scale=size,
            info_box_content=marker_info))
        return self

    @staticmethod
    def _markers_with_info(df_items, max_markers):
        marker_locs = df_items.iloc[:max_markers]
        info_box_template = \
            """
            <dl>            
            <dt>{description}</dt>
            </dl>
            """
        marker_info = [
            info_box_template.format(
                description=pprint.pformat(item_data.to_dict()))
            for _, item_data in marker_locs.iterrows()]

        return marker_locs, marker_info

    def draw_items(self, df_items=None, **kwargs):
        self.add_heatmap(df_items, **kwargs)
        self.add_markers(df_items, **kwargs)
        return self

    def write_html_to_file(self, path, title='exported map', map_height=800):
        for w in self.fig.widgets.values():
            if isinstance(w, ipywidgets.Layout) and str(w.height).endswith('px'):
                w.height = f'{map_height}px'

        ipywidgets.embed.embed_minimal_html(path, title=title, views=[self.fig])
        return self

    def write_html_to_str(self, title='exported map', map_height=800):
        with io.StringIO() as fp:
            self.write_html_to_file(fp, title=title, map_height=map_height)
            fp.flush()
            return fp.getvalue()

    @staticmethod
    def random_color():
        return tuple(map(int, np.random.randint(0, 255, 3)))  # because of bug in gmaps type checking

    @staticmethod
    def get_n_spaced_colors(n):
        # max_value = 16581375  # 255**3
        # interval = int(max_value / n)
        # colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        # return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

        HSV_tuples = [(x * 1.0 / n, 1.0, 0.8) for x in range(n)]
        RGB_tuples = list(map(lambda x:
                              tuple(list(map(lambda f: int(f * 255),
                                             colorsys.hsv_to_rgb(*x)))),
                              HSV_tuples))
        return RGB_tuples


class PropertyGeoMap(ItemsGeoMap):

    def __init__(self, link_base_url='www.domain.com.au', **kwargs):
        super().__init__(**kwargs)
        self.site_url = link_base_url

    def _markers_with_info(self, df_items, max_markers):
        marker_locs = df_items.iloc[:max_markers]
        marker_info = []
        for _, item_data in marker_locs.iterrows():
            item = item_data.to_dict()
            url = f"https://{self.site_url}/{item.get('property_id')}"
            marker_info.append(
                f"""     
                <dl><a style="font-size: 16px" href='{url}' target='_blank'>{url}</a><dt> 
                score: {item.get('score', np.nan) :.2f} | {item.get('price')} $ 
                {item.get('property_type')} ({item.get('buy_or_rent')}) | {item.get('bedrooms')} B ' \
                '| {item.get('bathrooms')} T | {item.get('carspaces')} P <br />  ' \
                '{item.get('land_area')} Sqm | in {item.get('suburb')} | with {item.get('features_list')}'</dt></dl>
                """)
        return marker_locs, marker_info


class RecommenderGeoVisualiser:

    def __init__(self,
                 recommender: BaseDFSparseRecommender,
                 items_handler: ItemsHandler,
                 link_base_url='www.domain.com.au'):
        self.recommender = recommender
        self.items_handler = items_handler
        self.mapper = ItemsGeoMapper(
            items_handler=self.items_handler,
            map=PropertyGeoMap(link_base_url=link_base_url))

    def random_user(self):
        return np.random.choice(self.recommender.all_users)

    def random_item(self):
        return np.random.choice(self.recommender.all_items)

    def _user_recommendations_and_scores(self, user):
        recos = self.recommender.get_recommendations([user])
        reco_items = np.array(recos[self.recommender._item_col].values[0])
        reco_scores = np.array(recos[self.recommender._prediction_col].values[0])
        reco_scores /= reco_scores.max()
        return reco_items, reco_scores

    def _similar_items_and_scores(self, item):
        if hasattr(self.recommender, 'get_similar_items'):
            simils = self.recommender.get_similar_items([item])
            simil_items = np.array(simils[self.recommender._item_col].values[0])
            simil_scores = np.array(simils[self.recommender._prediction_col].values[0])
            simil_scores /= simil_scores.max()
            return simil_items, simil_scores
        else:
            raise NotImplementedError(f"'get_similar_items' is not implemented / supported by "
                                      f"{self.recommender.__class__.__name__}")

    def _user_training_items(self, user):
        known_users = np.array([user])[~self.recommender.unknown_users_mask([user])]
        user_ind = self.recommender.user_inds(known_users)
        training_items = self.recommender.item_ids(
            self.recommender.train_mat[user_ind, :].indices).astype(str)
        return training_items

    def map_recommendations(self, user, path=None):
        training_items = self._user_training_items(user)
        reco_items, reco_scores = self._user_recommendations_and_scores(user)
        map_obj = self.mapper.map_recommendations(
            train_ids=training_items,
            reco_ids=reco_items,
            scores=reco_scores
        )
        if path:
            map_obj.write_html_to_file(path, title=f'user: {user}')
        else:
            return map_obj.write_html_to_str(title=f'user: {user}')

    def map_similar_items(self, item, path=None):
        simil_items, simil_scores = self._similar_items_and_scores(item)
        map_obj = self.mapper.map_similar_items(
            source_id=item,
            similar_ids=simil_items,
            scores=simil_scores
        )
        if path:
            map_obj.write_html_to_file(path, title=f'item: {item}')
        else:
            return map_obj.write_html_to_str(title=f'item: {item}')