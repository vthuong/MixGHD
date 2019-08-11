import numpy as np
import mpmath
import sympy
from scipy.spatial.distance import mahalanobis
from scipy.special import kv, kve
from sklearn.cluster import KMeans
from sklearn import mixture

class ClusterParam:
    def __init__(self):
        """
        Init
        params:
            mu: numpy array
                Center of Cluster
            alpha: numpy array
                skewness
            sigma: 2d array of size p x p
                Covariance matrix
            omega:
            lamda:

        return: None
        """
        self.mu = None
        self.alpha = None
        self.sigma = None
        self.omega = None
        self.lamda = None

    def ipar(self, data, wt, lc):
        """
            Initialize parameter
        """
        # Get dimensions of the data
        dims = data.shape

        # Initialize the wt if empty
        if wt is None:
            wt = np.ones(dims[0])

        self.mu = lc
        self.alpha = np.zeros(dims[1])

        if dims[1] == 1:
            self.sigma = np.var(data)
            self.sigma = self.sigma if self.sigma > 0.1 else 0.1
        else:
            cov = np.abs(np.diag(np.cov(m=data.T, aweights=wt, bias=True)))
            # adjust the var to 0.1 if the var is too small
            for i in range(dims[1]):
                cov[i] = cov[i] if cov[i] > 0.1 else 0.1

            self.sigma = np.diag(cov)

        # Initialize omega, lamda (default value as provided)
        self.omega = 1
        self.lamda = -1/2

    def __str__(self):
        return f'Mu\n{self.mu}\n\n' + \
               f'Alpha\n{self.alpha}\n\n' + \
               f'Sigma\n{self.sigma}\n\n' + \
               f'Omega\n{self.omega}\n\n' + \
               f'Lambda\n{self.lamda}\n'


class Gpar:
    IGPAR_MODEL_BASED = 0
    IGPAR_HIERARCHICAL = 1
    IGPAR_RANDOM = 2
    IGPAR_KMEDOIDS = 3
    IGPAR_KMEANS = 4

    def __init__(self):
        """
            Init G-Param collection object
            params:
                G: integer
                    number of clusters


        """
        self.cluster_list = None
        self.pi = 0

    def rgparGH(self, data, g=2, wt=None, l=None):
        """
            generate parameter
        """
        # Get dimensions of the data
        dims = data.shape

        # Initialize default value of weight matrix (all equal)
        # if not provided
        if wt is None:
            wt = np.ones((dims[0], g)) * (1.0 / g)

        # Initialize the weight
        self.cluster_list = [ClusterParam() for i in range(g)]
        for i in range(g):
            self.cluster_list[i].ipar(data, wt[:, i], l[i])

        self.pi = np.ones(g) * (1.0 / g)


def igpar3(data=None, g=None, n=10, method=Gpar.IGPAR_KMEDOIDS, nr=10):
    if g == 1:
        lc = np.array([np.mean(data, axis=0)])
        l = np.ones(data.shape[0])
        z = np.zeros((data.shape[0], g))
        combinewk(z, l)

        gpar = Gpar()
        gpar.rgparGH(data=data, g=g, wt=z, l=lc)

        # Only 10 iteration is allowed
        for j in range(n):
            try:
                EMgrstepGH(data=data, gpar=gpar, v=1, label=l, w=z)
            except:
                print("Exception occured in igpar3  g = 1")
        return gpar
    else:
        if method == Gpar.IGPAR_MODEL_BASED:
            gmm = mixture.GaussianMixture(n_components=g,
                                          covariance_type='full')
            gmm.fit(data)
            l = gmm.predict(data)
            lc = gmm.means_

            z = combinewk(weights=np.zeros(data.shape[0], g), label=l)
            gpar = Gpar()
            gpar.rgparGH(data=data, g=g, wt=z, label_center=lc)

            for j in range(n):
                try:
                    EMgrstepGH(data=data, gpar=gpar, v=1, label=l, w=z)
                except:
                    print("Exception occured in igpar3  g = 1")
            return gpar
        elif method == Gpar.IGPAR_KMEDOIDS:
            # I dont know the appropriate number for numlocal and maxneighbor so
            # I take it as 10 and 50
            clarans_ins = clarans(data=data, number_clusters=g,
                                  numlocal=10, maxneighbor=50)
            clarans_ins.process()
            lc = model.get_medoids()
            l = model.get_clusters()

            # Assume that lc and l are numpy array/ need to do coversion if not
            z = combinewk(weights=np.zeros(data.shape[0], g), label=l)
            gpar = Gpar()
            gpar.rgparGH(data=data, g=g, wt=z, l=lc)

            for j in range(n):
                try:
                    EMgrstepGH(data=data, gpar=gpar, v=1, label=l, w=z)
                except:
                    print("Exception occured in igpar3  kmedois")
            return gpar
        elif method == Gpar.IGPAR_KMEANS:
            # case of Kmeans
            kmeans_ins = KMeans(n_clusters=g)
            kmeans_ins.fit(data)
            l = kmeans_ins.predict(data) + 1
            lc = kmeans_ins.cluster_centers_
            z = np.zeros((data.shape[0], g))
            combinewk(z, l)

            gpar = Gpar()
            gpar.rgparGH(data=data, g=g, wt=z, l=lc)

            for j in range(n):
                try:
                    EMgrstepGH(data=data, gpar=gpar, v=1, label=l, w=z)
                except:
                    print("Exception occured in igpar3 kmeans")
            return gpar

# Using an initialization criterion to obtain cluster parameters. It is usually called when
# clustering is in execution. It is better than ipar because it involves EM Algorithm.
def igpar(data=None, g=None, method=Gpar.IGPAR_KMEDOIDS, nr=None):
    """
        using an initialization criterion to
        obtain cluster parameters. It is usually called when clustering is in execution, if not always. It is better
        than rgparGH, because it involves EM algorithms, although part of its execution also relies on
        rgparGH. However, it is a very awkward function, as the only thing it does is to call another function,
        igpar3, and return that function’s output
    """
    ##initialization
    gpar = igpar3(data=data, g=g, n=10, method=method, nr=nr)
    return gpar


def combinewk(weights=None, label=None):
    """
        Function to calculate the weighted matrix
        based on the data label
        Params:
        weights: numpy array
            matrix of size num_observations x num_labels
        label: numpy array
            array of label
        return
    """
    if label is None:
        print("Label cannot be None")
        return

    if label.size == 0:
        return

    for i in range(weights.shape[0]):
        if label[i] != 0:
            weights[i, :] = [0 if label[i] != j + 1 else 1
                             for j in range(weights.shape[1])]

def weightsGH(data, gpar, v=1):
    if not gpar:
        return np.zeros((data.shape[0], len(gpar.pi)))

    G = len(gpar.pi)
    if G > 1:
        zlog = np.zeros((data.shape[0], G))

        for k in range(G):
            zlog[:, k] = ddghypGH(data, gpar.cluster_list[k], log=True)

        w = np.zeros((G, data.shape[0]))
        wt = gpar.pi
        for k in range(data.shape[0]):
            z = zlog[k]
            fstar = v * (z + np.log(wt)) - max(v * (z + np.log(wt)))
            x = np.exp(fstar)
            if sum(x) == 0:
                x = np.ones(len(x))
            w[:, k] = x / sum(x)
        w = w.transpose()
    else:
        w = np.ones((data.shape[0], G))

    return w


def ddghypGH(x, par, log=False, invS=None):
    mu = par.mu
    sigma = par.sigma
    alpha = par.alpha

    if (len(mu) != len(alpha)):
        print('mu and alpha do not have the same length.')
        return

    d = len(mu)
    omega = np.exp(np.log(par.omega))
    lamda = par.lamda

    if not invS:
        invS = np.linalg.pinv(sigma)

    pa = omega + np.matmul(np.matmul(alpha, invS), alpha)
    delta = np.array([mahalanobis(x[i, :], mu, invS)**2
                      for i in range(x.shape[0])])
    mx = omega + delta

    kx = np.sqrt(mx * pa)
    xmu = x - mu

    lvx = np.zeros((x.shape[0], 4))
    lvx[:, 0] = (lamda - d/2) * np.log(kx)
    lvx[:, 1] = np.log(kve(lamda - d/2, kx)) - kx
    lvx[:, 2] = np.matmul(np.matmul(xmu, invS), alpha)

    lv = np.zeros(6)
    if np.log(np.linalg.det(sigma)) is np.inf: # need checking
        sigma = np.eye(len(mu))

    lv[0] = -1/2 * np.log(np.linalg.det(sigma))
    lv[0] -= d/2 * (np.log(2) + np.log(np.pi))
    lv[1] = omega - np.log(kve(lamda, omega))
    lv[2] = -lamda/2 * np.log(1)
    lv[3] = lamda * np.log(omega) * 0
    lv[4] = (d/2 - lamda) * np.log(pa)

    val = np.sum(lvx.transpose(), axis=0) + sum(lv)
    if not log:
        val = np.exp(val)

    return val

def log_besselKv(nu, y):
    return np.log(kve(nu, y)) - np.log(y)
# kve(x, y) would be equal to besselK(x = y, nu = x, expon.scaled = T)

def log_besselKvFA(nu, y):
    val = np.log(kv(nu, y))
    if np.isinf(val):
        val = np.log(sympy.Float(mpmath.besselk(nu, y)))
    return val

def weighted_sum(x, wt):
    return sum(x * wt)

def weighted_mean(x, wt):
    return weighted_sum(x, wt) / sum(wt)

def Rlam(x, lam=None):
    v1 = kve(lam + 1, x) * np.exp(x)
    v0 = kve(lam, x) * np.exp(x)
    val = v1 / v0

    if np.isinf(v1) or np.isinf(v0) or v0 == 0 or v1 == 0:
        lv1 = np.log(sympy.Float(mpmath.besselk(np.abs(lam + 1), x)))
        lv0 = np.log(sympy.Float(mpmath.besselk(np.abs(lam), x)))
        val = np.exp(lv1 - lv0)

    return val

def llikGH(data, gpar):
    G = len(gpar.pi)
    logz = np.zeros((data.shape[0], G))
    for k in range(G):
        logz[:, k] = ddghypGH(data, gpar.cluster_list[k], log=True)

    wt = gpar.pi
    val = 0
    for k in range(data.shape[0]):
        z = logz[k]
        val += np.log(sum(np.exp(z) * wt))
    return val

def update_ol(omega, lamda, ABC=None, n=1):
    """
        update omega and lambda. In most cases, n = 2. See page 10.
        To update omega and lambda, we need to maximize the function
        q_g(w_g, lambda_g) = ....

        ol lilely contains 2 components: a vector and a numerical value
    """
    for i in range(n):
        if (ABC[2] == 0):
            lamda = 0
        else:
            bv = richardson_gradient(log_besselKvFA, x = np.array([lamda]),
                                     y = np.array([omega]), eps = 1e-8,
                                     d = 0.0001, r = 6, v = 2)
            lamda = ABC[2] * (lamda / bv)

        Rp = Rlam(omega, lam= +lamda)
        Rn = Rlam(omega, lam= -lamda)

        f1 = Rp + Rn - (ABC[0] + ABC[1])
        f2 = (Rp ** 2 - (2 * lamda + 1) / omega * Rp - 1)
        f2 += (Rn ** 2 - (2 * (-lamda) + 1) / omega * Rn - 1)

        if (omega - f1 / f2 > 0):
            omega = omega - f1 / f2

    return (omega, lamda)

def update_mu_alpha_S_ol(x, par, weights=None, v=None, invS=None,
                         alpha_known=None):
    """
    Perform parameter estimation using EM framework described in
    Browne and McNicholas pg 9, 10

    Params:
        x: 2d array
            dataset
        par:  Cluster_Info
            single cluster parameter
        invS: 2d array
            inverse of covariance matrix

    """
    if weights is None:
        weights = np.ones(x.shape[0])

    if invS is None:
        invS = np.linalg.pinv(par.sigma)

    # expectations of w, 1/w, log(w) given x
    # w      : a = a_ig = E[W_ig | x_i, Z_ig = 1]
    # 1/w    : b = b_ig = E[1 / W_ig | x_i, Z_ig = 1]
    # log(w) : c = c_ig = E[log W_ig | x_i, Z_ig = 1]
    # abc is a MATRIX
    abc = gigGH(x=x, par=par, invS=invS)
    d = par.mu.shape[0]

    # Obtain n_g, a_g_bar, b_g_bar, and c_g_bar
    # ABC is a LIST
    # A = a_g_bar
    # B = b_g_bar
    # C = c_g_bar

    # Do the equivalent of th following in R
    sumw = np.sum(weights)
    # ABC = apply(abc, 2, weighted_sum, wt = weights) / sumw
    # basically scale each row of abc by the respective weight and sum each column
    ABC = [weighted_sum(abc[i], weights) / sumw for i in range(3)]

    if alpha_known is None:
        A = ABC[0]
        B = ABC[1]

        # abc[, 2] is b = b_ig = E[1 / W_ig | x_i, Z_ig = 1]
        # u is z_ig_hat * (b_g_bar - b_ig)
        # t is z_ig_hat * (a_g_bar * b_ig - 1)
        # See the formula for mu_g_hat and beta_g_hat
        u = (B - abc[1]) * weights
        t = (A * abc[1] - 1) * weights

        # the numerator of the formula mu_g_hat and beta_g_hat
        T = np.sum(t)

        # alpha.new is actually beta_g_hat, but recall the parameterization
        # beta = eta * alpha, where eta = 1, so calling alpha or beta would
        # be just the same

        mu_new = np.sum(np.multiply(x, t[:, np.newaxis]), axis=0) / T
        alpha_new = np.sum(np.multiply(x, u[:, np.newaxis]), axis=0) / T
    else:
        alpha_new = alpha_known
        mu_new = np.array([weighted_mean(x[:, i], abc[1] * weights)
                           for i in range(d)])
        mu_new -= alpha_new / ABC[1]

    alpha_new = alpha_new * v

    # TO DO: check the covariance
    A = wt_cov_ML(x, wt = abc[1] * weights, center = mu_new) * ABC[1]

    # x_g_bar
    r = [weighted_sum(x[:, i], weights) / sumw for i in range(d)]

    # This is the update covariance matrix
    # - (outer(r - mu.new, alpha.new) is -(x_g_bar - mu_g_hat) * (beta_g_hat)'
    # - outer(alpha.new, r - mu.new)) is - (beta_g_hat) * (x_g_bar - mu_g_hat)'
    # + outer(alpha.new,alpha.new) * ABC[1] is a_g_bar * beta_g_hat * (beta_g_hat)'
    R = A - (np.outer(r - mu_new, alpha_new) + np.outer(alpha_new, r - mu_new))
    R = R + np.outer(alpha_new, alpha_new) * ABC[0]

    for i in range(R.shape[1]):
        if R[i, i] < 0.00001:
            R[i, i] = 0.00001

    par.mu = mu_new
    par.alpha = alpha_new
    par.sigma = R
    par.omega, par.lamda = update_ol(par.omega, par.lamda, ABC = ABC, n = 2)

def gigGH(x=None, par=None, invS=None):
    """
        Calculate expectation of W, 1/W and log(W) (E-Step)
        W is assumed to follow Generalized Inverse Gaussian W~GIG(psi, chi, lambda)
        Refer to p9 A Mixture of Generalized Hyperbolic Distributions
        Params:
            a, v: numerical values
            b : numpy 2d array

        # Calculate a, b, c - expectations of w, 1/w, log(w) given x
        # w      : a = a_ig = E[W_ig | x_i, Z_ig = 1]
        # 1/w    : b = b_ig = E[1 / W_ig | x_i, Z_ig = 1]
        # log(w) : c = c_ig = E[log W_ig | x_i, Z_ig = 1]
        # abc is a MATRIX
        # returns a matrix with dim length(a) x 3 stima

        # a = omega_g + (beta_g)' * sigma_inverse * beta_g
        # b = omega + delta(x_i, mu_g | sigma_inverse)
        # kv1 is the denominator BesselK
        # kv is the numerator BesselK
        Input
        x: 2d array
            data
        par: ClusterParam object

        Return:
            tuple contains 3 values: E[W], E[1/W] and E[log(W)]
    """
    if invS is None:
        invS = np.linalg.inv(par.sigma)
    omega = par.omega

    a = omega + np.matmul(np.matmul(par.alpha, invS), par.alpha)
    delta = np.array([mahalanobis(x[i, :], par.mu, invS) ** 2
                      for i in range(x.shape[0])])
    b = omega + delta
    v = par.lamda - par.mu.shape[0] / 2

    sab = np.sqrt(a * b)
    kvv = kve(v + 1, sab) / kve(v, sab)
    sb_a = np.sqrt(b / a)

    # Expected value of w and 1/w
    w = kvv * sb_a
    invw = kvv / sb_a - 2 * v / b
    logw = np.log(sb_a) + richardson_gradient(log_besselKv,
                                              x = v * np.ones(sab.shape),
                                              y = sab, eps = 1e-8, d = 0.0001,
                                              r = 6, v = 2)
    return (w, invw, logw)

def mainMGHD(data, gpar0, G, n, label, eps, method, nr=None):
    if label is not None:
        label_list = np.unique(label[np.where(label > 0)])
        lc = np.array([np.mean(data[np.where(label == i)], axis=0)
                       for i in label_list])
        n_prelabel = len(label_list)
        if n_prelabel < G:
            km = KMeans(n_clusters=G)
            km.fit(data)
            km.predict(data)
            lc2 = km.cluster_centers_
            dist_mat = np.array(
                [[np.sqrt(np.sum((lc2[x, :] - lc[y, :]) ** 2))
                  for y in range(n_prelabel)]
                 for x in range(G)])
            for h in range(n_prelabel):
                min_index = np.argmin(dist_mat[:, h])
                dist_mat = np.delete(dist_mat, min_index, 0)
                lc2 = np.delete(lc2, min_index, 0)

            lc = np.concatenate(lc, lc2, axis=0)
        z = np.zeros((data.shape[0], G), 1 / G)
        combinewk(z, label)
        if gpar0 is None:
            gpar = Gpar()
            gpar.rgparGH(data, G, z, lc)
        else:
            gpar = gpar0
    else:
        if gpar0 is None:
            try:
                gpar = igpar(data, G, method, nr)
            except:
                print("Cluster parameters initialization failed")
        else:
            gpar = gpar0

    loglik = np.zeros(n)
    for i in range(3):
        try:
            EMgrstepGH(data, gpar, v=1, label=label)
        except:
            print("EM generation step failed")
        loglik[i] = llikGH(data, gpar)

    # while loop, run the exact
    # gpar = EMgrstepGH(data = data, gpar = gpar, v = 1, label = label)
    # loglik[i] = llikGH(data, gpar)
    # Run EMStepGH
    # i is equivalent to the counter in original R code of run_EMStepGH
    count = 2
    maxit = n
    while get_asym_loglik(loglik) > eps and count < (maxit - 1):
        count = count + 1  # temp[3]: count
        gpar = EMgrstepGH(data=data, gpar=gpar, v=1,
                          label=label)  # temp[2]: gpar
        loglik[count] = llikGH(data, gpar)  # temp[1]: loglik

    N = data.shape[0]
    pcol = data.shape[1]

    if (count < n):
        loglik = np.hstack((loglik[:count], loglik[n:]))

    BIC = 2 * loglik[count] # count - 1
    BIC -= np.log(N) * (
                (G - 1) + G * (2 * pcol + 2 + pcol * (pcol - 1) / 2))
    z = weightsGH(data=data, gpar=gpar)
    ICL = BIC + 2 * np.sum(np.log(np.max(z, axis=1)))
    AIC = 2 * loglik[i - 1]
    AIC -= 2 * ((G - 1) + G * (2 * pcol + 2 + pcol * (pcol - 1) / 2))
    AIC3 = 2 * loglik[i - 1] - 3 * (
            (G - 1) + G * (2 * pcol + 2 + pcol * (pcol - 1) / 2))

    return {'loglik': loglik, 'gpar': gpar, 'z': z,
            'map': MAPGH(data=data, gpar=gpar, label=label),
            'BIC': BIC, 'ICL': ICL, 'AIC': AIC, 'AIC3': AIC3}

def EMgrstepGH(data=None, gpar=None, v=1, label=None, w=None):
    if w is None:
        try:
            w = weightsGH(data, gpar, v)
        except:
            pass

    if label is not None:
        combinewk(w, label)

    G = len(gpar.pi)
    for k in range(G):
        update_mu_alpha_S_ol(data, gpar.cluster_list[k],
                             w[:, k], 1, None, None)
    gpar.pi = sum(w) / data.shape[0]

def get_asym_loglik(loglik):
    n = len(loglik)
    if n < 3:
        print("Must have at least 3 likelihood values")
        return

    lk_plus_1 = loglik[n - 1]
    lk = loglik[n - 2]
    lk_minus_1 = loglik[n - 3]
    ak = (lk_plus_1 - lk) / (lk - lk_minus_1)

    lk_plus_1_Inf = lk + (lk_plus_1 - lk) / (1 - ak)
    val = lk_plus_1_Inf - lk
    val = 0 if np.isnan(val) else val
    val = 1 if val < 0 else val
    return val

def MAPGH(data, gpar, label=None):
    w = weightsGH(data, gpar, v=1)
    if label is not None:
        combinewk(w, label)
    return np.apply_along_axis(lambda row: row.argmax() + 1, axis=1, arr=w)

def richardson_gradient(f, x, y=None, eps=1e-4, d=0.0001,
                        zero_tol=np.finfo(np.float32).eps, r=4, v=2):
    """
    #------------------------------------------------------------------------
    # 1 Applying Richardson Extrapolation to improve the accuracy of
    #   the first and second order derivatives. The algorithm as follows:
    #
    #   --  For each column of the derivative matrix a,
    #	  say, A1, A2, ..., Ar, by Richardson Extrapolation, to calculate a
    #	  new sequence of approximations B1, B2, ..., Br used the formula
    #
    #	     B(i) =( A(i+1)*4^m - A(i) ) / (4^m - 1) ,  i=1,2,...,r-m
    #
    #		N.B. This formula assumes v=2.
    #
    #   -- Initially m is taken as 1  and then the process is repeated
    #	 restarting with the latest improved values and increasing the
    #	 value of m by one each until m equals r-1
    #
    # 2 Display the improved derivatives for each
    #   m from 1 to r-1 if the argument show.details=T.
    #
    # 3 Return the final improved  derivative vector.
    #-------------------------------------------------------------------------
          x is the points we need to find derivative at
          f is in the form of (nu, x)
    """
    n = x.shape[0]
    a = np.empty((r, n))
    h = np.abs(d * x) + eps * (abs(x) < zero_tol)

    for k in range(r):
        a[k] = (f(x + h, y) - f(x - h, y)) / (2 * h)
        h = h / v

    for m in range(1, r):
        a = (a[1:(r + 1 - m)] * (4 ** m) - a[0:(r - m)]) / (4 ** m - 1)

    return np.squeeze(a)

def wt_cov_ML(data, wt, center):
    if center.shape[0] != data.shape[1]:
        print('Length of center must equal the number of columns')
        return

    x = np.apply_along_axis(lambda col: col * np.sqrt(wt / sum(wt)),
                            axis = 0, arr = data - center)
    return x.transpose() @ x