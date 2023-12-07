import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import signal
import numba as nb

def dtw_average(signals, average_signal, iterations, window_size):
    nsignals = len(signals)
    paths = []
    match_indices = []
    signal_indices =[]

    for j in range(iterations):

        for n in range(nsignals):
            #perform dtw between signals and average
            paths.append(pruned_dtw(average_signal, signals[n], window_size)[1])

            #extract paired template indexes
            match_indices.append(np.array(paths[n])[:,0])

            #extract the paired signal indexes
            signal_indices.append(np.array(paths[n])[:,1])

        #length of template
        n_average = len(average_signal)

        #initialize array to store the new average signal
        average_signal = []
        time_index = []

        #iterate through indices of the template signal
        for i in range(n_average):

            for n in range(nsignals):

                #check to see if the template index i was paired with indexes of signaln
                if i in match_indices[n]:


                    #if so find indexes of pairs which contain the template index
                    indices = np.where(match_indices[n] == i)


                    #check to see if signal_values exists
                    if 'signal_values' in locals():

                        #if so find the paired signal2 indices, extract signal values and concat to signal_values
                        signal_values = np.concatenate((signal_values,signals[n][signal_indices[n][tuple(indices)]]))

                    else:
                        #if not find the paired signal2 indices, extract signal values
                        signal_values = signals[n][signal_indices[n][tuple(indices)]]

            #if signal_values exists take the mean and update average_signal[i]
            if 'signal_values' in locals():
                average_signal.append(np.mean(signal_values))
                del signal_values
                #save index value for time vector
                time_index.append(i)
        #convert lists to numpy arrays
        average_signal = np.array(average_signal)
        time_index = np.array(time_index)

    return time_index, average_signal

def warped_signal(path, warp_index, signal1):

    """the path array is an array of tuples of length 2 pairing indices from the two signals
    the warped index is the either 0 or 1 depending on which signal is desired to be warped. The
    values from the tuple correspond to the time indices of the warped signal
    """
    #create the warped_index and time_index arrays
    if warp_index == 0:

        warp_index = np.array(path)[:,0]
        time_index = np.array(path)[:,1]

    else:
        warp_index = np.array(path)[:,1]
        time_index = np.array(path)[:,0]

    #unique values of from the time_index array
    unique_time_index = np.unique(time_index)

    #instantiate warped_signal array
    warped_signal = np.zeros(len(unique_time_index))

    #create the time vector for the warped signal
    #time = time[unique_time_index]
    i =0

    #take the mean of warped signal values which share the same time_index
    for val in np.nditer(unique_time_index):
        indices = np.where(time_index == val)
        warped_signal[i] = np.mean(signal1[warp_index[tuple(indices)]])
        i += 1

    return unique_time_index, warped_signal

def purity_score(nclusters, data_labels, clusters):
    in_cluster_count= {}
    correspondence = {}
    cluster_labels = np.unique(clusters)
    N = len(data_labels)
    for k in range(len(cluster_labels)):
        cluster_label = cluster_labels[k]
        cluster = np.where(clusters ==cluster_label)
        if np.any(cluster):
            data_labels = np.array(data_labels, dtype='int')

            in_cluster_count[cluster_label] = np.bincount(data_labels[cluster]).max()
            correspondence[cluster_label] = np.bincount(data_labels[cluster]).argmax()
    purity = 0
    for val in in_cluster_count.values():
        purity = purity + val/N

    return correspondence, purity

def tkeo_operator(data, k = 1):

    npnts = len(data[0])
    nsignals = len(data)
    filt_data = deepcopy(data)
    for i in range(nsignals):
        for n in range(k, npnts-k):
            filt_data[i][n] = data[i][n]**2-data[i][n-1]*data[i][n+1]
    return filt_data

def standard_scaler(data):
    nsignals = len(data)
    for i in range(nsignals):
    #normalize to avoid spikes/plateaus during dtw due to differences in amplitudes
        normed_sig = (data[i]- np.mean(data[i]))/np.std(data[i])
        data[i] = normed_sig
    return data


def dtw_distance_matrix(data, window_size, indices):
    nsignals = len(data)
    dtw_dists = [[] for _ in range(nsignals)]
    for i in range(nsignals):

        signal1 = data[i][indices[i][0]:indices[i][1]]
        for j in range(nsignals):
            signal2 = data[j][indices[j][0]:indices[j][1]]

            if i <= j:
                cost_matrix, path, distance = pruned_dtw(signal1, signal2, window_size)

                dtw_dists[i].append(distance)

    for i in range(nsignals):
        dtw_dists[i].extend([dtw_dists[j][i] for j in range(i+1, nsignals)])
    dtw_dists = np.array(dtw_dists)
    return dtw_dists

def dtw_correlation(data, window_size, indices):
    nsignals = len(data)
    dtw_corrs = np.zeros((nsignals,nsignals))
    for i in range(nsignals):

        signal1 = data[i][indices[i][0]:indices[i][1]]
        for j in range(nsignals):
            signal2 = data[j][indices[j][0]:indices[j][1]]

            if i == j:
                dtw_corrs[i,j] = 1

            if dtw_corrs[i,j] == 0:
                cost_matrix, path, distance = pruned_dtw(signal1, signal2, window_size)

                #correlation between warped signal i and signal j
                index_w, warped_s = warped_signal(path, 0, signal1)
                corr = np.corrcoef(warped_s, signal2[(index_w)])[0,1]
                dtw_corrs[i,j] = corr

                #correlation between signal i and warped signal j
                index_w, warped_s = warped_signal(path, 1, signal2)
                corr = np.corrcoef(warped_s, signal1[(index_w)])[0,1]
                dtw_corrs[j,i] = corr
            else:
                break

    return dtw_corrs

def pruned_dtw(matched, warped, window_size):
    #create distance matrix
    N = len(matched)
    M = len(warped)
    ###
    ub_coordinate_list, fraction, ub_partials = upper_bound_partials(matched, warped)
    ###
    #initialize auxiliary pruning variables
    start_column = 1
    end_column = 1

    #window must be greater than N-M
    window_size = max(window_size, N-M)

    #create cost matrix
    cost_matrix = np.ndarray((N+1, M+1))

    #initialize to infinity
    cost_matrix[:] = np.inf

    cost_matrix[0,0] = 0
    ###
    UB = ub_partials[1]
    ###
    #create traceback matrix
    traceback_matrix = np.ones((N,M))*np.inf

    #initialize window elements to zero
    for i in range(1,N+1):

        if N>=M:
            ub_col_index = int(np.floor(i*fraction))

        else:
            ub_col_index = int(np.floor(i/fraction))


        beg = max(start_column,ub_col_index-window_size)
        end = min(M+1, ub_col_index+window_size+1)
        smaller_found = False
        end_column_next = ub_col_index

        for j in range(beg, end):
            cost = np.abs(matched[i-1]-warped[j-1])

            penalty = [cost_matrix[i-1,j-1],  #match 0
                       cost_matrix[i-1,j], # insertion 1
                       cost_matrix[i, j-1]] # deletion 2
            penalty_index = np.argmin(penalty)
            traceback_matrix[i-1,j-1] = penalty_index
            cost_matrix[i,j] = cost +penalty[penalty_index]

            ###
            #if i == ub_col_index and i != N:
            if (i,j) in ub_coordinate_list:
                ub_index = ub_coordinate_list.index((i,j))
                UB = cost_matrix[i,j] + ub_partials[ub_index]
            ###

            ##pruning algorithm
            if cost_matrix[i,j] > UB:
                if smaller_found == False:
                    start_column = j+1

                if j >= end_column:
                    break

            else:
                smaller_found = True
                end_column_next = j+1
        end_column = end_column_next
    #traceback from bottom right corner

    i = N-1
    j = M-1

    path = [(i, j)]

    while i>0 and j>0:

        tb_type = traceback_matrix[i,j]

        if tb_type == 0:
            #match
            i = i-1
            j = j-1

        elif tb_type == 1:
            #insertion
            i = i-1
        else:
            #deletion
            j = j-1

        path.append((i,j))

    #strip infinite row and column from cost matrix
    cost_matrix = cost_matrix[1:, 1:]

    distance = cost_matrix[-1,-1]
    #invert path
    path= path[::-1]

    return cost_matrix, path, distance

def upper_bound_partials(signal1, signal2):
    signal_list = [signal1, signal2]
    coordinate_list = [[],[]]
    signal_lengths = [len(x) for x in signal_list]

    if signal_lengths[0]!= signal_lengths[1]:
        index_long = np.argmax(signal_lengths)
        index_short = np.argmin(signal_lengths)
        long_len = len(signal_list[index_long])
        short_len = len(signal_list[index_short])
        fraction = short_len/long_len
        long_index_list = [i for i in range(long_len)]
        short_index_list = [int(np.floor(x*fraction)) for x in range(long_len)]
        warped_short = np.take(signal_list[index_short], short_index_list)
        ub_partials = np.abs(signal_list[index_long]-warped_short)[::-1]
        coordinate_list[index_long] = long_index_list
        coordinate_list[index_short] = short_index_list

    else:
        ub_partials = np.abs(signal_list[0]-signal_list[1])[::-1]
        fraction = 1
        index_list1 = [i for i in range(signal_lengths[0])]
        index_list2 = [i for i in range(signal_lengths[1])]
        coordinate_list[0] = index_list1
        coordinate_list[1] = index_list2

    ub_partials = np.cumsum(ub_partials)[::-1]
    coordinate_list = zip(coordinate_list[0],coordinate_list[1])
    return list(coordinate_list), fraction, ub_partials

def rolling_std(signal1, k):
    n = len(signal1)
    window_size = 2*k+1
    std_ts = np.zeros(n)
    for ti in range(n):
        low_bnd = max(0, ti-k)
        up_bnd = min(n,ti+k)
        tmp_sig = signal1[low_bnd:up_bnd]
        std_ts[ti] = np.std(tmp_sig)

    return std_ts

def segment_indices(signal1, thresh, k, pad):
    halfwin = k
    std_ts = rolling_std(signal1, halfwin)
    n = len(signal1)
    indices = np.where(std_ts>thresh)[0][[0,-1]]
    indices[0] = max([indices[0]-pad, 0])
    indices[1] = min([indices[1]+pad, n])
    return indices

def segment_data(data, thresh, k, pad):
    nsignals = len(data)
    indices = np.zeros((nsignals,2),dtype = int)
    segmented_data = []
    for i in range(nsignals):
        ind = segment_indices(data[i], thresh = thresh, k = k, pad = pad)
        #segmented_data.append(data[i][indices[0]:indices[1]])
        indices[i] = ind
    return indices

def tsne_visualization(dtw_dists, centroids, data_labels):
    model = TSNE(learning_rate = 20, perplexity = 3)
    nsignals = len(dtw_dists)
    #fit model
    transformed = model.fit_transform(np.concatenate([dtw_dists, centroids]))

    xs = transformed[:,0]
    ys = transformed[:,1]

    for i in range(4):
        indices = np.where(data_labels == i)

        if i == 1:
            plt.scatter(xs[indices],ys[indices], marker = 'o', label = 'sine waves')
        if i == 2:
            plt.scatter(xs[indices],ys[indices], marker = '^', label = 'sawtooth waves')
        if i == 3:
            plt.scatter(xs[indices],ys[indices], marker = 's', label = 'square waves')

    plt.scatter(xs[nsignals:],ys[nsignals:], marker = 'p', label = 'centroids')

    plt.title('TSNE of DTW correlation matrix')
    plt.legend()
    # plt.show()

def representative_signal_indexes(clusters, dtw_dists):
    #find optimal ingroup representative of cluster
    unique_cluster_labels = np.unique(clusters)

    best_signals_indxs = np.zeros([len(unique_cluster_labels)], dtype = int)
    cluster_indxs_list = []
    i=0

    for label in unique_cluster_labels:

        cluster_indxs = np.where(clusters ==label)[0]
        cluster_indxs_list.append(cluster_indxs)
        ingroup_corrs = dtw_dists[:, cluster_indxs]
        ingroup_corrs = ingroup_corrs[cluster_indxs,:]

        #columnwise averages
        averages = np.mean(ingroup_corrs, axis = 0)
        idx = np.argmax(averages)
        best_signal = cluster_indxs[idx]
        best_signals_indxs[i] = best_signal
        i = i+1

    return cluster_indxs_list, best_signals_indxs

def preprocessing(data, thresh = 10, k = 10, pad = 10):
    #standard scaler
    filt_data = standard_scaler(data)

    #tkeo operator
    filt_data = tkeo_operator(data)

    #standard scaler
    filt_data = standard_scaler(filt_data)
    print(filt_data)

    thresh = 0.9
    k=10
    pad = 10

    indxs = segment_data(filt_data, pad = pad, thresh = thresh, k=k)

    return indxs, filt_data

def average_incluster_signals(filt_data, indxs, clusters, dtw_dists):
    cluster_indxs_list, best_signals_indxs = representative_signal_indexes(clusters, dtw_dists)
    iterations = 10
    window_size = 30
    average_signal_list = []
    time_index_list = []
    for j in range(len(cluster_indxs_list)):

        best_sig_indx = best_signals_indxs[j]
        group_indexes = cluster_indxs_list[j]
        average_signal = filt_data[best_sig_indx][indxs[best_sig_indx][0]:indxs[best_sig_indx][1]]
        signals_list = [filt_data[i][indxs[i][0]:indxs[i][1]] for i in group_indexes]

        time_index, average_signal = dtw_average(signals_list, average_signal, iterations, window_size)
        average_signal_list.append(average_signal)
        time_index_list.append(time_index)
    return time_index_list, average_signal_list

def predict_test_data(test_filt_data, test_indxs,average_signal_list, window_size):
    nsignals = len(test_filt_data)
    nclasses = len(average_signal_list)

    predicted_labels = np.zeros([nsignals])
    for i in range(nsignals):
        pred_list = []

        signal1 = test_filt_data[i][test_indxs[i][0]:test_indxs[i][1]]
        for j in range(nclasses):
            signal2 = average_signal_list[j]
            cost_matrix, path, distance = pruned_dtw(signal1, signal2, window_size)
            dist = cost_matrix[-1,-1]
            pred_list.append(dist)

        #map to original data_labels
        predicted_label = correspondence[np.argmin(pred_list)]
        predicted_labels[i] = predicted_label
    return predicted_labels

def load_data(filename, nsignals=8, down_sampling=None, short=None):
    import pandas as pd

    df = pd.read_csv(filename, sep="\t", header=None)
    data_labels = df.loc[:,0].to_numpy(dtype=np.int32)
    data = df.loc[:,1:].to_numpy(dtype=np.float32)

    if isinstance(down_sampling, int):
        new_data = []

        for i in range(0, data.shape[1] // down_sampling):
            # print(data[:, i*down_sampling:(i+1)*down_sampling].shape)
            x = np.sum(data[:, i*down_sampling:(i+1)*down_sampling], axis=1, keepdims=True) / down_sampling
            print(x.shape)
            new_data.append(x)
        new_data = np.concatenate(new_data, axis=1, dtype=np.float32)
        # print(new_data)
        data = new_data
        print(data[0,:])

    if isinstance(short, int):
        data = data[:, :short]

    time = np.linspace(0,1,data.shape[1])
    print(data.shape, data_labels.shape, time.shape)
    return time, data, data_labels


# time, data, data_labels = generate_data(nsignals = 30)
# d = input("down_sampling") or None
d = None
time, data, data_labels = load_data("train_data.tsv", down_sampling=d)

nsignals = len(data)
for n in range(nsignals):
    plt.plot(time, data[n,:]+5*n)
# plt.show()


# #standard scaler
filt_data = standard_scaler(data)

# #tkeo operator
filt_data = tkeo_operator(data)

# #standard scaler
filt_data = standard_scaler(filt_data)

for n in range(nsignals):
    plt.plot(time, filt_data[n,:]+10*n)
# plt.show()


indxs = segment_data(filt_data, pad = 10, thresh = 0.99, k=10)

for i in range(nsignals):
    plt.plot(time[indxs[i][0]:indxs[i][1]],filt_data[i][indxs[i][0]:indxs[i][1]]+10*i)
# plt.show()


dtw_dists = dtw_correlation(data = filt_data, window_size = 30, indices = indxs)

# #kmeans clustering
kmeans_model = KMeans(n_clusters = 3)
clusters = kmeans_model.fit_predict(dtw_dists)

# #calculate purity score
correspondence, purity = purity_score(3, data_labels, clusters)
print('purity score: {}'.format(purity))

# tsne_visualization(dtw_dists, kmeans_model.cluster_centers_, data_labels)

time_index_list, average_signal_list = average_incluster_signals(filt_data, indxs, clusters, dtw_dists)

# for i in range(len(average_signal_list)):
#     if correspondence[i] == 1:
#         label = 'sine'
#     elif correspondence[i]==2:
#         label = 'sawtooth'
#     else:
#         label = 'square'
#     plt.plot(time_index_list[i], average_signal_list[i]+10*i, label = label)
# plt.legend()
# plt.title('Average Cluster Signals')
# plt.show()


# time, test_data, data_labels = generate_data(nsignals = 100)
time, test_data, data_labels = load_data("test_data.tsv")
test_indxs, test_filt_data = preprocessing(test_data)

predicted_labels = predict_test_data(test_filt_data, test_indxs,average_signal_list, 30)
_, purity = purity_score(3, data_labels, predicted_labels)
print('purity score: {}'.format(purity))


