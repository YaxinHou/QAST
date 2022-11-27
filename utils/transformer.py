import numpy as np
import pandas as pd

import torch
from torch.nn import functional

import warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')


class DataTransformer(object):

    def __init__(self, n_clusters=8, epsilon=0.01):
        self.n_clusters = n_clusters
        self.epsilon = epsilon

    # train bayesian gmm for continuous column
    def _fit_continuous(self, column, data):
        gm = GaussianMixture(n_components=8, covariance_type='full', tol=1e-3,
                             reg_covar=1e-6, max_iter=100, n_init=3, init_params='kmeans',
                             weights_init=None, means_init=None, precisions_init=None,
                             random_state=42, warm_start=False,
                             verbose=0, verbose_interval=10)
        gm.fit(data)
        components = gm.weights_ > self.epsilon
        num_components = components.sum()
        return {
            'name': column,
            'model': gm,
            'components': components,
            'output_info': [(1, 'tanh'), (num_components, 'softmax')],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, column, data):
        ohe = OneHotEncoder(sparse=False)
        ohe.fit_transform(data).reshape(-1, 1)
        categories = ohe.n_values_[0]

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    # fit gmm for continuous columns and one hot encoder for discrete columns
    def fit(self, data):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.dtypes = data.infer_objects().dtypes
        self.meta = []
        for column in data.columns:
            column_data = data[[column]].values
            meta = self._fit_continuous(column, column_data)
            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self.meta.append(meta)

    def _transform_continuous(self, column_meta, data):
        components = column_meta['components']
        model = column_meta['model']
        means = model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self.n_clusters))
        features = (data - means) / stds
        probs = model.predict_proba(data)

        n_opts = components.sum()
        features = features[:, components]
        probs = probs[:, components]

        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            opt_sel[i] = np.argmax(probs[i], axis=0)
        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])

        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        return [features, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        print(data.shape)
        s = encoder.transform(data)
        print("encoding:", s.shape)
        return encoder.transform(data)

    # take raw data and output a matrix data
    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.meta:
            column_data = data[[meta['name']]].values
            values += self._transform_continuous(meta, column_data)

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_continuous(self, meta, data, sigma):
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -1000
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * std_t + mean_t
        return column

    def _inverse_transform_discrete(self, meta, data):
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    # take matrix data and output raw data
    # output uses the same type as input to the transform function
    def inverse_transform(self, data, sigmas):
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas else None
                inverted = self._inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = []
                for i in columns_data:
                    i = list(i)
                    inverted.append(np.argmax(i))
            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)
        if not self.dataframe:
            output = output.values

        return output

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], hard=True, tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)
