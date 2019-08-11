import numpy as np
import pandas as pd

from sklearn import mixture
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv('irik.csv', sep=',', header=None)
irik = df.values
X = irik[:, 0:2]

gmm = mixture.GaussianMixture(n_components=5, covariance_type='full')
gmm.fit(X)
gmm_label = gmm.predict(X)

# True label: irik[:, 2]
adjusted_rand_score(irik[:, 2], gmm_label)

# Label center: lc
gmm.means_


