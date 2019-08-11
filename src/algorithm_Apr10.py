import numpy as np
import GPar_UpdateMar15

def dGHD(x, p, mu, alpha, sigma, omega = 1, lamda = 0.5, log = False):
    if not mu:
        mu = np.zeros(p)
    if not alpha:
        alpha = np.zeros(p)
    if not sigma:
        sigma = np.eye(p, p)

    if len(mu) != len(alpha):
        print("mu and alpha do not have the same length")
        return

    d = len(mu)
    omega = np.exp(np.log(omega))
    invS = np.linalg.inv(sigma)

    pa = omega + np.cross(np.cross(alpha, invS), alpha)
    delta = np.array([mahalanobis(x[i, :], mu, invS)**2 for i in range(x.shape[0])])
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
    lv[0] = -1 / 2 * np.log(np.linalg.det(sigma))
    lv[0] -= d / 2 * (np.log(2) + np.log(np.pi))
    lv[1] = omega - np.log(kve(lamda, omega))
    lv[2] = -lamda/2 * np.log(1)
    lv[3] = lamda * np.log(omega) * 0
    lv[4] = (d/2 - lamda) * np.log(pa)

    val = np.sum(lvx.transpose(), axis=0) + sum(lv)
    if not log:
        val = np.exp(val)

    return val

def get_modes(array):
    if type(array) is not np.ndarray:
        print("Array-like required")
        return

    unique_values, freq = np.unique(array, return_counts=True)
    return unique_values[np.where(freq == np.amax(freq))]

def search_expand_grid(modes, optimal_lbl, cur_lbl, ptr):
    """
    modes:  list
        label options to be considered
    max_pos: list
        position of the optimal label
        the last value is number of unique value in current set
    cur_pos: list
        current position of the label options (from each class)
    ptr: position of list currently considered
    """
    if ptr == len(modes):
        n = len(set(cur_lbl))
        if n > optimal_lbl[-1]:
            optimal_lbl[-1] = n
            optimal_lbl[:ptr] = cur_lbl
    else:
        for j in range(len(modes[ptr])):
            cur_lbl[ptr] = modes[ptr][j]
            search_expand_grid(modes, optimal_lbl, cur_lbl, ptr + 1)
        

def get_cluster_labels(mat):
    if type(mat) is not np.ndarray:
        print("2-way array required")
        return

    G = mat.shape[0]
    modes = [0 for i in range(G)]
    for i in range(G):
        modes[i] = get_modes(mat[i])
    
    # search for optimal labeling
    optimal_lbl = [0] * (G + 1)
    cur_lbl = [0] * G
    ptr = 0
    search_expand_grid(modes, optimal_lbl, cur_lbl, ptr)
    
    # The set above may contains duplications
    # Replace the duplicated value with a value that not appear yet
    if (optimal_lbl[-1] == G):
        return optimal_lbl[:-1]
    else:
        rst = [-1] * G
        options = [(i + 1) for i in range(G)]
        
        flag = True
        while(flag):
            flag = False
            
            # Remove all list with single unique value
            for i in range(G):
                if rst[i] == -1 and len(modes[i]) == 1:
                    if modes[i] in options:
                        # Raise flag
                        rst[i] = modes[i][0]
                        options.remove(modes[i][0])
                        flag = True
                i = i + 1
            # Update the options on all other clusters.
            # Remove the label in the previous step.
            for i in range(G):
                modes[i] = list(set(modes[i]) & set(options))
        
        # If we cannot determine the label of one cluster. Assign its label to be
        # one of its options left or a random value if all options are not feasible.
        for i in range(G):
            if rst[i] == -1:
                if optimal_lbl[i] in options:
                    rst[i] = optimal_lbl[i]
                    options.remove(optimal_lbl[i])
                else:
                    rst[i] = options[0]
                    options = options[1:]
        
        return rst
    
OVERLAP_SILHOUETTE = 0
OVERLAP_MAPPING = 1

IGPAR_MODEL_BASED = 0
IGPAR_HIERARCHICAL = 1
IGPAR_RANDOM = 2
IGPAR_KMEDOIDS = 3

MODELSEL_BIC = 0
MODELSEL_ICL = 1
MODELSEL_AIC3 = 2
MODELSEL_AIC = 3

def do_active_label(data, sentences, label_list, pre_label, label,
                    how_many = [5, 20], top_sensity = 0.2,
                    overlap_detection = OVERLAP_SILHOUETTE, diff_threshold = 0.05,
                    max_iter = 100, eps = 0.01, method = IGPAR_KMEDOIDS, nr = 10, modelSel = MODELSEL_BIC):
    # Label a numeric dataset by actively selecting a number of examples to
    # consult experts
    #
    # Arguments:
    #   data: a numeric data frame converted from the text documents to label.
    #         No NA is allowed in the data frame.
    #
    #   sentences: a vector of the original text documents.
    #
    #   label.list: possible possible labels for the text documents 
    #
    #   how.many: a vector of two numbers. The first number is how many 
    #             observations around each cluster's mode to sample, while 
    #             the second number is how many observations to sample from
    #             overlap areas (if any). Two default numbers, 5 and 20,
    #             are based the results of simulations
    #
    #   top.density: a number from 0 to 1 that indicates the percentage of 
    #                highest-density observations for sampling around the
    #                modes. It defines how large the sampling area around 
    #                each mode should be.
    #   
    #   overlap.detection: what overlap detection techniques should be used:
    #                      "silhouette" for Silhouette distance, and "mapping"
    #                      for mapping probability.
    #
    #   difference.threshold: if overlap.detection = "mapping", specifies how
    #                         much difference between the two highest mapping
    #                         probabilties of an observation, so that it is
    #                         considered lying in one of overlap areas.
    #   max.iter: (optional) the maximum number of iterations each EM algorithm 
    #             is allowed to use.
    #   
    #   eps: (optional) the epsilon value for the convergence criteria used in 
    #        the EM algorithms
    #
    #   method: a string that indicates intialization criterion. If not specified,
    #           K-medoids is used. Alternative methods are: hierarchical 
    #           "hierarchical", random "random", and model based "modelBased".
    #   nr: (optional) the number of starting values when method = "random". By
    #       default, nr = 10.
    #
    #   modelSel: (optional) a string indicating the model selection criterion, if
    #             not specified, BIC is used. Alternative methods are: AIC, ICL and
    #             AIC3.
    #
    # Return:
    #   A list of two S4 objects of class MixGHD with slots: index, BIC, ICL, AIC,
    #   AIC3, gpar, loglik, map and z. They are the models at the end of the first
    #   and second classifcations respectively.
    """
      if (missing(data)) {
    stop("data is required")
  }
  if (missing(sentences)) {
    stop("Original text documents are required")
  }
  if (missing(label.list)) {
    stop("All possible labels must be specified")
  }
  if (!is.data.frame(data)) {
    stop("data must be data frame")
  }
  if (sum(sapply(data, is.na)) > 0) {
    stop("No NA is allowed in data")
  }
  if (sum(sapply(data, is.numeric)) < ncol(data)) {
    stop("Columns of data must be all numeric")
  }
  if (!is.character(sentences)) {
    stop("sentences must be character")
  }
  if (!is.character(label.list)) {
    stop("label.list must be character")
  }
    """
    G = len(set(label_list))
    if G < 2:
        print("At least two distinct labels level required")
    
    """
      if (!is.vector(how.many) | any(is.na(how.many)) | !is.numeric(hoinw.many) |
      length(how.many) != 2 | any(how.many != floor(how.many))) {
    stop("how.many must be a vector of whole numbers with length 2 and no NA")
  }
  if (!is.numeric(top.density) | length(top.density) != 1 | 
      top.density < 0 | top.density > 1) {
    stop("top.density must be a number between 0 and 1")
  }
      
        overlap.detection <- match.arg(overlap.detection)
  if (overlap.detection == "mapping") {
    if (!is.numeric(difference.threshold) | length(difference.threshold) != 1 | 
      difference.threshold < 0 | difference.threshold > 1) {
      stop("difference.threshold must be a number between 0 and 1")
    }
  }
    """

    if label is not None:
        #labelset = list(set(label))
        prelabel = [0 if (x is None or x == 0) else label_list.index(x) for x in label]
        current_mod = MGDH(data = data, G = G, max_iter = max_iter, eps = eps, label = prelabel, 
                       method = method, nr = nr, modelSel = modelSel, scale = False)
    else:
        current_mod = MGDH(data = data, G = G, max_iter = max_iter, eps = eps, 
                       method = method, nr = nr, modelSel = modelSel, scale = False)
    
    nrow = data.shape(0)
    expert_labels = np.empty(nrow)
    available_indices = np.arange(1, nrow + 1)
    indices_to_ask = None # need to create sth iterable
    
    for label in set(current_mod.mapp):
        # Obtain observations in this cluster that expert haven't seen yet
        indices_in_cluster = np.where(np.equal(current_mod.mapp == label))
        indices_in_cluster = np.intersect1d(indices_in_cluster, available_indices)
        
        #Not sure
        cluster_gpars = current_mod.gpar[label]
        # TO DO: are we using CPL or Omega and lambda?
        densities = dGHD(data[indices_in_cluster], data.shape[1], cluster_gpars.mu,
                         cluster_gpars.alpha, cluster_gpar.sigma, cluster_gpar.cpl[0],
                         cluster_gpar.cpl[1])
        incicies_in_cluster = indicies_in_cluster[np.sort(densities)]
        
        # sample observations around the mode
        # sample.pool <- round(top.density * nrow(data[indices.in.cluster, ]))
        sample_pool = np.round(top_density * sum(indices_in_cluster))
        if sample_pool < how_many[0]:
            if sample_pool > 0:
                print("\nNot enough observations to sample. Returned as many as possible.\n")
                new_indicies = indices_in_cluster[:sample_pool]
            else:
                print("\nNo data to sample. Returned none.\n")
                new_indicies = None
        else:
            new_indicies = np.random.choice(indices_in_cluster[:sample_pool], how_many[0])
            
        
        # Add those observations to the expert consulting list
        indices_to_ask = np.hstack(indices_to_ask, new_indicies)
        available_indices = np.setdiff1d(available_indices, new_indicies)
    
    # Consult experts for the labels of sampled observations
    # This step will coerce expert.labels to character
    print ("Please indicate the appropriate label for the following sentences")
    for i in range(len(label_list)):
        print(i, " : ", label_list[i])
    print()
    
    for idx in indices_to_ask:
        print (sentences[idx])
        invalid_label = True
        while invalid_label:
            x = input("Label")
            if x <= 0 or x >= len(label_list):
                print ("The label input is invalid. Please select one from the list")
                for i in range(len(label_list)):
                    print(i, " : ", label_list[i])
                print()
            else:
                expert_labels[idx] = x
                invalid_label = False
        
        # 
    
    # Sort new labels obtained from experts in alphabetical order and map them
    # into sequential integers starting from 1. For example,
    # Finance, Marketing, Finance, Finance, Accounting will become 2, 3, 2, 2, 1
    # In the end, expert.labels will be converted back into numeric.
    label_org = np.sort(np.unique(expert_labels[indices_to_ask]))
    for i in indices_to_ask:
        expert_labels[i] = np.in1d(expert_labels[i], label_org)
    
    # Decide the label for each cluster, as observations sampled from the same
    # cluster do not necessarily have the same label. 
    un <- expert.labels[indices.to.ask].reshape(G, how_many[1])
    modes <- getClusterLabels(un)
    
    # Map each set of estimated parameters to its right cluster.
    # They will become the initial parameters of the first classification model. 
    gpar = current_mod.gpar
    for i in range(G):
        gpar.cluster_list[modes[i]] = current_mod.gpar.cluster_list[i]
        gpar.pi = current_mod.gpar.pi
    
    # Perform and save the first classification
    current_mod = MGHD(data, gpar, G, max_iter, eps, expert_labels, method, nr, modelSel)
    mods[0] = current_mod
    
    
    ######################### SAMPLE IN OVERLAP AREAS #########################
    # indices_to_ask =
    
    # Detect overlap areas using the specified technique
    indices_overlapping = None
    if overlap_detection == OVERLAP_SILHOUETTE:
        distance_matrix = euclidean_distances(data)
        silhouette_width = silhouette_score(distance_matrix, current_mod.mapp)
        indices_overlapping = np.where(np.less_equal(silhouette_width[:, 2]))
    else:
        # Mapping
        probability_matrix = current_mod.z.reshape(nrow, G)
        is_overlapping = np.apply_along_axis(lambda(x, threshold): return ((x[-1] - x[-2] <= threshold), 
                                                    1, np.sort(x), diff_threshold)
        indices_overlapping <- np.where(is_overlapping)
    # Sample in the overlap areas and make sure that experts won't have to see
    # them again
    
    indices_overlapping = np.intersect1d(available_indices, indices_overlapping)
    if len(indices_overlapping) > how_many[1]:
        indices_to_ask = np.random.sample(indices_overlapping, how_many[1])
    else:
        print ("\nNot enough observations to sample. Returned ", length(indices.to.ask))
        indices_to_ask = indices_overlapping
    
    available_indices = np.setdiff1d(available_indices, indices_to_ask)
    
    # Consult experts for the labels of sampled observations.
    # This step will coerce expert.labels to character
    print ("Please indicate the appropriate label for the following sentences")
    for i in range(len(label_list)):
        print(i, " : ", label_list[i])
    print()
    
    for idx in indices_to_ask:
        print (sentences[idx])
        invalid_label = True
        while invalid_label:
            x = input("Label")
            if x <= 0 or x >= len(label_list):
                print ("The label input is invalid. Please select one from the list")
                for i in range(len(label_list)):
                    print(i, " : ", label_list[i])
                print()
            else:
                expert_labels[idx] = x
                invalid_label = False
        
        
     # Sort new labels obtained from experts in alphabetical order and map them
     # into sequential integers starting from 1. For example,
     # Finance, Marketing, Finance, Finance, Accounting will become 2, 3, 2, 2, 1
     # In the end, expert.labels will be converted back into numeric.
     label_org = np.sort(np.unique(expert_labels[indices_to_ask]))
     for i in indices_to_ask:
        expert_labels[i] = np.in1d(expert_labels[i], label_org)
    
    gpar = current_mod.gpar
    current_mod = MGHD(data, gpar, G, max_iter, eps, expert_labels, method, nr, modelSel)
    mods[1] = current_mod
        
    return mods
    
    
mat = np.array([[5, 5, 3, 3, 5],
                [5, 4, 5, 4, 1],
                [5, 1, 2, 4, 1],
                [1, 3, 5, 3, 3],
                [3, 1, 3, 5, 3]])