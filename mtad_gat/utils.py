# code based on https://github.com/srigas/MTAD-GAT-mlflow/blob/main/utils.py
# with minor adjustments

import os
import numpy as np
import pandas as pd
import tqdm
import datetime
import mlflow

from torch.utils.data import DataLoader, Dataset, Subset
from math import floor, log
from scipy.optimize import minimize


def get_data(dataset, mode="train", start=0, end=None):
    """Get data to be used for training/validation/evaluation

    :param mode: train or eval, to get train or eval data
    :param start: starting index of dataset if not all data are to be used
    :param end: ending index of dataset if not all data are to be used
    """
    
    dataset_folder = os.path.join("datasets", dataset)

    # Load the data
    # WARNING: For good evaluation/inference, a total of window_size data need to be taken
    # from the train dataset and placed before the new data. This should not be done within
    # the model, but as a different pre-processing procedure.
    try:
        data = np.loadtxt(os.path.join(dataset_folder, f"{mode}.csv"),
                            delimiter=",", dtype=np.float32)[start:end, :]
        if mode=="train":
            # train data do not have labels - unsupervised learning
            labels = None
        else:
            labels = np.loadtxt(os.path.join(dataset_folder, f"{mode}_labels.csv"),
                                delimiter=",", dtype=np.float32)[start:end]
                                
    except (KeyError, FileNotFoundError):
        raise Exception("Only acceptable modes are train and eval.")

    return (data, labels)


class SlidingWindowDataset(Dataset):
    """Class that creates a sliding window dataset for a given time-series

    :param data: time-series data to be converted
    :param window_size: size of the sliding window
    :param stride: the number of different timestamps between two consecutive windows
    :param horizon: the number of timestamps for future predictions
    """
    def __init__(self, data, window_size, stride=1, horizon=1):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.horizon = horizon

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.window_size]
        y = self.data[start_index + self.window_size : start_index + self.window_size + self.horizon]
        return x, y

    def __len__(self):
        return len(range(0, len(self.data) - self.window_size - self.horizon + 1, self.stride))


def create_data_loader(dataset, batch_size, val_split=0.1, shuffle=True):
    """Create torch data loaders to feed the data in the model

    :param dataset: torch dataset
    :param batch_size: size of data batches
    :param val_split: if set to a non-zero value, an extra loader is created with val_split*100%
                      of the whole data, usually to be used for validation
    :param shuffle: wether to shuffle data and get random indices or not
    """
    if val_split is None:
        # Corresponds to the case of eval data or training without validation
        print(f"The size of the dataset is: {len(dataset)} sample(s).")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        extra_loader = None

    else:
        # Corresponds to the case with train/val splitting
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        extra_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"The size of the dataset is: {len(train_indices)} sample(s).")
        print(f"Reserved {len(val_indices)} sample(s) for validation.")

    return loader, extra_loader


def get_run_id(run_name, experiment_name):
    """Transform an input run_name to the run_id

    :param run_name: the input run_name to be transformed
    :param experiment_name: the name of the experiment in which to look for runs
    """

    # If no run_name is given, the last run is retrieved
    if run_name is None:
        run_name = "-1"
    else:
        run_name = str(run_name)

    # get corresponding experiment (using experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id

    # If run_name is given as a relative value (-1, -2, etc.), get the actual name
    if run_name.startswith('-'):
        
        runs = mlflow.search_runs(experiment_ids=exp_id)
        run_names = runs["tags.mlflow.runName"].values.tolist()
        date_times = [datetime.datetime.strptime(rn, '%d%m%Y_%H%M%S') for rn in run_names]
        date_times.sort()
        model_datetime = date_times[int(run_name)]
        run_name = model_datetime.strftime('%d%m%Y_%H%M%S')
    
    # Given the actual name, retrieve the run id
    run = mlflow.search_runs(experiment_ids=exp_id, filter_string=f'tags."mlflow.runName" = "{run_name}"')
    run_id = run['run_id'][0]

    return run_id


# ------------------------ THRESHOLD UTILITIES ------------------------------


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    Code from https://github.com/NetManAIOps/OmniAnomaly

    Attributes
    ----------
    proba : float
            Detection level (risk), chosen by the user

    extreme_quantile : float
            current threshold (bound between normal and abnormal events)

    data : numpy.array
            stream

    init_data : numpy.array
            initial batch of observations (for the calibration/initialization step)

    init_threshold : float
            initial threshold computed during the calibration step

    peaks : numpy.array
            array of peaks (excesses above the initial threshold)

    n : int
            number of observed values

    Nt : int
            number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor
        Parameters
        ----------
        q
                Detection level (risk)

        Returns
        ----------
        SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (
                    r,
                    100 * r / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t extreme quantile : %s\n" % self.extreme_quantile
                s += "Algorithm run : No\n"
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object

        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
                initial batch to calibrate the algorithm

        data : numpy.array
                data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print("The initial data cannot be set")
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
        ----------
        data : list, numpy.array, pandas.Series
                data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        level : float
                (default 0.98) Probability associated with the initial threshold t
        verbose : bool
                (default = True) If True, gives details about the batch initialization
        verbose: bool
                (default True) If True, prints log
        min_extrema bool
                (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ", end="")

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print("[done]")
            print("\t" + chr(0x03B3) + " = " + str(g))
            print("\t" + chr(0x03C3) + " = " + str(s))
            print("\tL = " + str(l))
            print("Extreme quantile (probability = %s): %s" % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                scalar function
        jac : function
                first order derivative of the function
        bounds : tuple
                (min,max) interval for the roots search
        npoints : int
                maximum number of roots to output
        method : str
                'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
                possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                observations
        gamma : float
                GPD index parameter
        sigma : float
                GPD scale parameter (>0)
        Returns
        ----------
        float
                log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
                numerical parameter to perform (default : 1e-8)
        n_points : int
                maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
                gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
                GPD parameter
        sigma : float
                GPD parameter
        Returns
        ----------
        float
                quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        """
		Run SPOT on the stream

		Parameters
		----------
		with_alarm : bool
			(default = True) If False, SPOT will adapt the threshold assuming \
			there is no abnormal values
		Returns
		----------
		dict
			keys : 'thresholds' and 'alarms'

			'thresholds' contains the extreme quantiles and 'alarms' contains \
			the indexes of the values which have triggered alarms

		"""
        if self.n > self.init_data.size:
            print(
                "Warning : the algorithm seems to have already been run, you \
            should initialize before running again"
            )
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile:
                    # if we want to alarm, we put it in the alarm list
                    if with_alarm:
                        alarm.append(i)
                    # otherwise we add it in the peaks
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        # self.peaks = self.peaks[1:]
                        self.Nt += 1
                        self.n += 1
                        # and we update the thresholds

                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    # self.peaks = self.peaks[1:]
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {"thresholds": th, "alarms": alarm}


def pot_threshold(init_score, score, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return threshold: pot result threshold
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    #print(f'While running POT, detected {len(ret["alarms"])} and calculated {len(ret["thresholds"])} thresholds.')

    pot_th = np.mean(ret["thresholds"])
    return pot_th


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def json_to_numpy(path):
    """Opens a .json artifact and casts its values as a numpy array
    :param path: path to look for the json artifact 
    """

    data = mlflow.artifacts.load_dict(path)

    npfile = np.asarray(list(data.values())).flatten()

    return npfile


def update_json(uri, name, new_data):
    """Opens a .json artifact and updates its contents with new data
    :param path: path to look for the json artifact
    :param new_data: dictionary that contains the new contents as key-value pairs    
    """
    try:
        data = mlflow.artifacts.load_dict(uri+"/"+name)
        data.update(new_data)
    except mlflow.exceptions.MlflowException:
        data = new_data

    mlflow.log_dict(data, name)


# ------------------------ EVALUATION UTILITIES ------------------------------


def get_metrics(y_pred, y_true):
    """Function to calculate metrics, given a predictions and an actual list of 0s and 1s.
    :param y_pred: list of 0s and 1s as predicted by the model
    :param y_true: list of 0s and 1s as ground truth anomalies
    """
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)

    TP = np.sum(y_pred * y_true)
    FP = np.sum(y_pred * (1 - y_true))
    FN = np.sum((1 - y_pred) * y_true)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return f1, precision, recall


def anoms_to_indices(anom_list):
    """Function that returns indices of anomalous values
    :param anom_list: list of 0s and 1s
    """
    ind_list = [i for i, x in enumerate(anom_list) if x == 1]
    xs = list(range(len(anom_list)))
    return ind_list, xs


def create_anom_range(xs, anoms):
    """Function that creates ranges of anomalies
    :param xs: list of indices to be used for the plot, auto generated by anoms_to_indices
    :param anoms: indices that belong in xs and correspond to anomalies
    """
    anomaly_ranges = []
    for anom in anoms:
        idx = xs.index(anom)
        if anomaly_ranges and anomaly_ranges[-1][-1] == idx-1:
            anomaly_ranges[-1] = (anomaly_ranges[-1][0], idx)
        else:
            anomaly_ranges.append((idx, idx))
    return anomaly_ranges


def PA(y_true, y_pred):
    """Function that performs the point-adjustment strategy
    :param y_true: list of 0s and 1s as ground truth anomalies
    :param y_pred: list of 0s and 1s as predicted by the model, so that they can be point-adjusted
    """
    new_preds = np.array(y_pred)

    # Transform into indices lists
    y_true_ind, xs = anoms_to_indices(y_true)
    y_pred_ind, _ = anoms_to_indices(y_pred)

    # Create the anomaly ranges
    anom_ranges = create_anom_range(xs, y_true_ind)

    # Iterate over all ranges
    for start, end in anom_ranges:
        itms = list(range(start,end+1))
        # if we find at least one identified instance
        if any(item in itms for item in y_pred_ind):
            # Set the whole event equal to 1
            new_preds[start:end+1] = 1

    return new_preds


def calculate_latency(y_true, y_pred):
    """Function that calculates the latency of all events' prediction
    :param y_true: list of 0s and 1s as ground truth anomalies
    :param y_pred: list of 0s and 1s as predicted by the model, so that they can be point-adjusted
    """
    events = []
    identified_events = []
    
    # Identify separate events in the ground truth
    start = None
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if start is None:
                start = i
        elif start is not None:
            events.append((start, i-1))
            start = None
    
    # Identify events and calculate delays in predictions
    for event in events:
        start, end = event
        delay = None
        for i in range(start, end+1):
            if y_pred[i] == 1:
                delay = i - start
                break
        if delay is not None:
            identified_events.append((event, delay))
    
    num_correct = len(identified_events)
    total_delay = sum(delay for _, delay in identified_events)
    avg_delay = total_delay / num_correct if num_correct > 0 else 0
    
    # Events not identified
    not_identified_events = [event for event in events if event not in [e[0] for e in identified_events]]
    
    return num_correct, avg_delay, identified_events, not_identified_events