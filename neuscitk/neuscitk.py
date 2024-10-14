'''
Neuroscience toolkit
Written for Python 3.12.4
@ Jeremy Schroeter, August 2024
'''

import os
os.environ['OMP_NUM_THREADS'] = '1'
import errno
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt
from subprocess import PIPE, run
from scipy.io import loadmat
from scipy.signal import lfilter, butter, filtfilt, dimpulse, find_peaks, freqz, convolve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class LabChartDataset:
    '''
    Dataset class for organizing and interfacing with LabChart data that
    has been exported as a MATLAB file.

    Parameters
    ----------
    file_path : str
        The path to the LabChart data file.
    '''
    def __init__(self, file_path: str):
        if os.path.exists(file_path) is False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
        
        self.matlab_dict = loadmat(file_name=file_path)
        self.n_channels = len(self.matlab_dict['titles'])
        
        self.data = {f'Channel {ch + 1}' : self._split_blocks(ch) for ch in range(self.n_channels)}

    
    def _split_blocks(self, channel: int) -> list[np.ndarray]:
        '''
        Private method fo building the data dictionary
        '''

        # LabChart concatenates channels for some reason so this is a workaround
        raw = self.matlab_dict['data'].reshape(-1)
        channel_starts = self.matlab_dict['datastart'][channel] - 1
        channel_ends = self.matlab_dict['dataend'][channel]

        n_blocks = channel_starts.shape[0]
        channel_blocks = []

        for idx in range(n_blocks):
            start = int(channel_starts[idx])
            end = int(channel_ends[idx])
            channel_blocks.append(raw[start:end])
        
        return channel_blocks



    def get_block(self, indices: list[int] | int) -> dict[np.ndarray]:
        '''
        Given a block index or list of block indices, returns the data for each channel
        during that block.

        Parameters
        ----------
        indices : list[int] | int
            The block index or list of block indices to retrieve.
        
        Returns
        -------
        dict[np.ndarray]
            A dictionary of blocks. Each block contains the data for each channel like (n_channel, length_of_block).
        '''

        # If only one block is requested, return block as an array
        if isinstance(indices, int):
            if 0 == indices:
                raise ValueError('Block indices are 1-based and cannot be 0.')
            
            block_data = []
            for channel in range(self.n_channels):
                block_data.append(self.data[f'Channel {channel + 1}'][indices - 1])
            if self.n_channels == 1:
                return np.array(block_data)[0]
            return np.array(block_data)


        # If multiple blocks are requested, return a dictionary of blocks
        if 0 in indices:
            raise ValueError('Block indices are 1-based and cannot be 0.')

        data_to_fetch = {}
        for block in indices:
            block_data = []
            for channel in range(self.n_channels):
                block_data.append(self.data[f'Channel {channel + 1}'][block - 1])
            
            data_to_fetch[f'block_{block}'] = np.array(block_data)
        
        return data_to_fetch
    

    def organize_by_pages(self, page_map: dict) -> None:
        '''
        Organizes the data into pages based on the page map.
        
        Parameters
        ----------
        page_map : dict
            A dictionary that maps page names to the block indices that belong to that page.
        
        Returns
        -------
        None
        '''

        self.pages = {page : self.get_block(indices) for page, indices in page_map.items()}
    

    def get_page(self, page_name: str) -> dict[np.ndarray]:
        '''
        Retrieves the data for a specific page.

        Parameters
        ----------
        page_name : str
            The name of the page to retrieve.
        
        Returns
        -------
        dict[np.ndarray]
            A dictionary of blocks. Each block contains the data for each channel like (n_channel, length_of_block).
        '''

        return self.pages[page_name]
    

    def concat_blocks(self, blocks: list[int]) -> np.ndarray:
        '''
        Concatenates blocks of data.
        
        Parameters
        ----------
        blocks : list[int]
            The blocks to concatenate.
            
        Returns
        -------
        np.ndarray
            The concatenated data.
        '''

        blocks = self.get_block(blocks)
        return np.hstack([block for block in blocks.values()])


    @property
    def fs(self) -> float | np.ndarray:
        '''
        Returns the sampling frequency of the data. If sampleiung frequency is constant, returns a float.
        '''
        fs = self.matlab_dict['samplerate']

        if np.all(fs == fs[0]):
            return fs.reshape(-1)[0]
        else:
            return fs

class Filter:
    '''
    Class for applying low-, high-, and band-pass filters to 1d signals.

    Parameters
    ----------
    fs: int
        The sampling frequency of the signal.

    lowcut : float | None, optional
        Lowcut frequency for a high or band pass filter. Leave as none
        if implementing low pass. Default is None.

    highcut : float | None, optional
        Highcut frequency for a low or band pass filter. Leave as none
        if implementing high pass. Default is None.

    order : int, optional
        The order of the filter. Default is 4.
    '''

    def __init__(
            self,
            fs: int,
            lowcut: float | None = None,
            highcut: float | None = None,
            order: int = 4
    ) -> None:
        
        if lowcut is not None and highcut is not None:
            if lowcut > highcut:
                raise ValueError('Lowcut frequency cannot be greater than highcut frequency.')
            if highcut < lowcut:
                raise ValueError('Highcut frequency cannot be less than lowcut frequency.')
            
            self.lowcut = lowcut
            self.highcut = highcut
            self.b, self.a = butter(N=order, Wn=[lowcut, highcut], btype='bandpass', fs=fs)
        
        elif lowcut is not None and highcut is None:
            self.lowcut = lowcut
            self.b, self.a = butter(N=order, Wn=lowcut, btype='highpass', fs=fs)
        
        elif lowcut is None and highcut is not None:
            self.highcut = highcut
            self.b, self.a = butter(N=order, Wn=highcut, btype='lowpass', fs=fs)
        
        else:
            raise ValueError('Either lowcut or highcut frequency must be specified.')
        
    def apply(self, arr: np.ndarray) -> np.ndarray:
        '''
        Applies the filter to the input signal.
        
        Parameters
        ----------
        arr : np.ndarray
            The input signal to filter.

        Returns
        -------
        np.ndarray
            The filtered signal.
        '''
        if not isinstance(arr, np.ndarray):
            raise TypeError('Input signal must be a numpy array.')
    
        return filtfilt(self.b, self.a, arr)
    
    @property
    def impulse_response(self) -> np.ndarray:
        '''
        The impulse response of the filter.
        '''
        system = (self.b, self.a, 1)
        _, h = dimpulse(system, n=100)
        return h[0].flatten()
    
    @property
    def frequency_response(self) -> tuple[np.ndarray]:
        '''
        The frequency response of the filter.
        '''
        w, h = freqz(self.b, self.a)
        return w, h


class FiringRateConverter:
    '''
    Class for converting spike trains to firing rates. To see the equations
    used to calculate the firing rate, see Dayan, Abbott 2001, pgs. 11-14.

    Parameters
    ----------
    fs : int
        The sampling frequency of the spike train.
    
    filter_type : str, {'gaussian', 'exponential', 'boxcar'}, optional
        The type of filter to use. Default is 'gaussian'.
    
    time_constant : float, optional
        The time constant of the filter. Default is 0.05.

    '''

    def __init__(self, fs: int, filter_type: str = 'gaussian', time_constant: float = 0.05):

        # Input validation
        if fs <= 0:
            raise ValueError('Sampling frequency must be greater than 0.')
        if time_constant <= 0:
            raise ValueError('Time constant must be greater than 0.')
        if filter_type not in ['gaussian', 'exponential', 'boxcar']:
            raise ValueError('Filter type must be either "gaussian", "exponential", or "boxcar".')


        self.fs = fs
        self.filter_type = filter_type
        self.time_constant = time_constant
        self._create_filter_kernel()
    
    
    def _build_gaussian_kernel(self) -> np.ndarray:
        '''
        Private method for creating the Gaussian filter kernel.
        '''
        n = int(self.time_constant * self.fs * 5)
        t = np.arange(0, n) / self.fs
        kernel = np.exp((-t**2) / (2 * self.time_constant**2))
        return kernel / np.sum(kernel)

    
    def _build_exponential_kernel(self) -> np.ndarray:
        '''
        Private method for creating the exponential filter kernel.
        '''
        n = int(self.time_constant * self.fs * 5)
        t = np.arange(0, n) / self.fs
        kernel = (1 / self.time_constant)**2 * t * np.exp(-t / self.time_constant)
        return kernel / np.sum(kernel)
    
    def _build_boxcar_kernel(self) -> np.ndarray:
        '''
        Private method for creating the boxcar filter kernel.
        '''
        n = int(self.time_constant * self.fs)
        kernel = np.ones(n)
        return kernel / n

    def _create_filter_kernel(self) -> np.ndarray:
        '''
        Private method for creating the filter kernel.
        '''
        if self.filter_type == 'gaussian':
            self.kernel = self._build_gaussian_kernel()
        
        elif self.filter_type == 'exponential':
            self.kernel = self._build_exponential_kernel()
        
        elif self.filter_type == 'boxcar':
            self.kernel = self._build_boxcar_kernel()
        

    def apply(self, spike_train: np.ndarray) -> np.ndarray:
        '''
        Applies the filter to the spike train.
        '''

        if not isinstance(spike_train, np.ndarray):
            raise TypeError('Input signal must be a numpy array.')

        firing_rate = lfilter(self.kernel, [1], spike_train)
        return firing_rate * self.fs


class SortedSpikes:
    '''
    Object for interacting with the results of running the spike sorter.
    This object should only be returned by the spike sorter function and is not
    to be directly created by the user.

    Parameters
    ----------
    sort_summary: dict
        The summary of the spike sorting results. Output of the spike sorter function.
    '''

    def __init__(self, sort_summary: dict):

        # Load the spike sorting results
        self.sorted_spikes = {cluster+1: cluster_data for cluster, cluster_data in sort_summary['clusters'].items()}
        self.params = sort_summary['parameters']
        self.pca_embeddings = sort_summary['pca_embeddings']
        self.pca_var_explained = sort_summary['pca_var_explained']
        self.labels = sort_summary['cluster_labels']

        self._raw_data = sort_summary['raw_data']
        self._spike_times = sort_summary['spike_times']
        self._waveforms = sort_summary['waveforms']
    
    
    def get_cluster_waveforms(self, cluster: int) -> np.ndarray:
        '''
        Retrieves the waveforms for a specific cluster.

        Parameters
        ----------
        cluster : int
            The cluster number to retrieve.

        Returns
        -------
        np.ndarray
            The waveforms for the specified cluster.
        '''
        return self.sorted_spikes[cluster]['waveforms']


    def get_cluster_spike_times(self, cluster: int) -> np.ndarray:
        '''
        Retrieves the spike times for a specific cluster.

        Parameters
        ----------
        cluster : int
            The cluster number to retrieve.

        Returns
        -------
        np.ndarray
            The spike times for the specified cluster.
        '''
        return self.sorted_spikes[cluster]['spike_times']
    
    def get_spike_train(self, cluster: int) -> np.ndarray:
        '''
        Retrieves the spike train for a specific cluster.
        
        Parameters
        ----------
        cluster : int
            The cluster number to retrieve.
            
        Returns
        -------
        np.ndarray
            The spike train for the specified cluster.
        '''
        spike_train = np.zeros_like(self._raw_data)
        cluster_spike_times = self.sorted_spikes[cluster]['spike_times']
        spike_train[cluster_spike_times] = 1
        return spike_train
    

    def plot_clusters(self) -> None:
        '''
        Plot all waveforms colored by cluster, and their position in PC space
        '''

        fig = plt.figure(figsize=(10, 5))
        pca_ax = fig.add_subplot(121)
        wf_ax = fig.add_subplot(122)

        for cluster_id in np.unique(self.labels):
            cluster_waveforms = self.pca_embeddings[self.labels == cluster_id]
            pca_ax.scatter(
                cluster_waveforms[:, 0],
                cluster_waveforms[:, 1],
                label=f'Cluster {cluster_id}'
            )
        
        pca_ax.set_xlabel('PC1')
        pca_ax.set_ylabel('PC2')
        pca_ax.legend()

        # Iterate over clusters and plot the waveforms
        for idx, (cluster, cluster_data) in enumerate(self.sorted_spikes.items()):
            waveforms = cluster_data['waveforms']

            mean_waveform = np.mean(waveforms, axis=0)

            for jdx, wav in enumerate(waveforms):
                if jdx == 0:
                    wf_ax.plot(
                        wav,
                        color = f'C{idx}',
                        alpha = 0.2,
                        label = f'Cluster {cluster}'
                    )
                else:
                    wf_ax.plot(
                        wav,
                        color = f'C{idx}',
                        alpha = 0.2
                    )
            wf_ax.plot(
                mean_waveform,
                c='black',
                lw=3
            )
        
        wf_ax.set_xlabel('time (samples)')
        wf_ax.set_ylabel('amplitude (V)')
        wf_ax.legend()
        fig.tight_layout()
        plt.show()


    
    def shift_clusters(self, cluster: int, shift: int) -> None:
        '''
        Shifts the spikes times of a cluster by a specific amount
        
        Parameters
        ----------
        cluster : int
            The cluster to shift.
            
        shift : int
            The number of timepoints to shift the spikes by.
        '''
        self.sorted_spikes[cluster]['spike_times'] -= shift

        window_size = self.params['waveform_window']
        new_waveforms = []

        for spike in self.sorted_spikes[cluster]['spike_times']:
            new_waveforms.append(self._raw_data[spike - window_size:spike + window_size])

        self.sorted_spikes[cluster]['waveforms'] = np.vstack(new_waveforms)
        self._waveforms = np.vstack([self.sorted_spikes[cluster]['waveforms'] for cluster in self.sorted_spikes.keys()])
        self._spike_times = np.hstack([self.sorted_spikes[cluster]['spike_times'] for cluster in self.sorted_spikes.keys()])

        # Remove duplicate rows from waveforms and duplicate elements from spike times
        unique_waveforms, unique_indices = np.unique(self._waveforms, axis=0, return_index=True)
        self._waveforms = unique_waveforms
        self._spike_times = self._spike_times[unique_indices]


        new_embeddings = PCA().fit_transform(self._waveforms)
        self.pca_embeddings = new_embeddings
        self.hand_pick_clusters()

    
    def _regorganize_clusters(self, new_labels: np.ndarray) -> None:
        '''
        Private method for reorganizing clusters based on new labels.
        '''
        self.sorted_spikes = {}
        for idx, cluster in enumerate(np.unique(new_labels)):
            self.sorted_spikes[cluster] = {
                'spike_times' : self._spike_times[new_labels == cluster],
                'waveforms' : self._waveforms[new_labels == cluster]
            }
        self.labels = new_labels
        self._spike_times 


    def hand_pick_clusters(self) -> None:
        '''
        Function for handpicking clusters with a lasso selector if
        you are unhappy with the fit from KMeans
        '''
        
        # Create file paths where we will store temporary data
        os.mkdir(os.path.join(os.getcwd(), 'temp'))
        embeddings_path = os.path.join(os.getcwd(), 'temp', 'embeddings.json')
        waveforms_path = os.path.join(os.getcwd(), 'temp', 'waveforms.json')
        new_labels_path = os.path.join(os.getcwd(), 'temp', 'new_labels.json')

        # Create tempfiles
        with open(embeddings_path, 'w') as f:
            json.dump(self.pca_embeddings.tolist(), f)
        
        with open(waveforms_path, 'w') as f:
            json.dump(self._waveforms.tolist(), f)
        
        # Run the hand picking script
        with importlib.resources.path('neuscitk', 'hand_pick_clusters.py') as path:
            path = os.path.join(path)
        
        # Run script as a subprocess
        run(
            [
                'python',
                path,
                embeddings_path,
                waveforms_path,
                new_labels_path
            ],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            check=True
        )

        # Load and return newly chosen clusters
        with open(new_labels_path, 'r') as f:
            new_labels = json.load(f)
        new_labels = np.array(new_labels)

        # Delete temp files
        os.remove(embeddings_path)
        os.remove(waveforms_path)
        os.remove(new_labels_path)
        os.rmdir(os.path.join(os.getcwd(), 'temp'))

        # Change state to reflect newly selected clusters
        self._regorganize_clusters(new_labels)


def apply_band_pass(
        arr: np.ndarray,
        fs: int,
        lowcut: float,
        highcut: float,
        order: int = 4
) -> np.ndarray:
    return Filter(fs=fs, lowcut=lowcut, highcut=highcut, order=order).apply(arr)


def moving_average(arr: np.ndarray, fs: int, window_size: float) -> np.ndarray:
    '''
    Calculate the moving average of an array

    Parameters
    ------------
    arr : np.ndarray
        The array to calculate the moving average of

    fs : int
        The sample rate of the array

    window_size : float
        The size of the window to use for the moving average

    Returns
    ------------
    np.ndarray
        The moving average of the input array
    '''
    n = int(fs * window_size)
    window = np.ones(n) / n
    return convolve(arr, window, mode='same')


def moving_std(arr: np.ndarray, fs: int, window_size: float) -> np.ndarray:
    '''
    Calculate the moving standard deviation of an array

    Parameters
    ------------
    arr : np.ndarray
        The array to calculate the moving standard deviation of

    fs : int
        The sample rate of the array

    window_size : float
        The size of the window to use for the moving standard deviation

    Returns
    ------------
    np.ndarray
        The moving standard deviation of the input array
    '''
    ma = moving_average(arr, fs, window_size)
    ma_sq = moving_average(arr ** 2)
    return np.sqrt(ma_sq - ma**2)

def detect_spikes(arr: np.ndarray, fs: int, window_size: int, multiplier: float = 4):
    '''
    Detect spikes in an array using an adaptive threshold

    Parameters
    ------------
    arr : np.ndarray
        The array to detect spikes in

    fs : int
        The sample rate of the array

    window_size : int
        The size of the window to use for the moving average and standard deviation

    multiplier : float
        The multiplier to use for the threshold

    Returns
    ------------
    np.ndarray
        The indices of the spikes in the input array
    '''
    ma = moving_average(arr, fs, window_size)
    mstd = moving_std(arr, fs, window_size)
    threshold = ma + mstd * multiplier
    peaks, _ = find_peaks(arr, height=threshold)
    return peaks


def extract_waveforms(arr: np.ndarray, fs: int, window_size: int, ma_window: float = 0.025, threshold_multiplier: float = 4) -> np.ndarray:
    '''
    Extract spike waveforms from an array
    
    Parameters
    ------------
    arr : np.ndarray
        The array to extract waveforms from

    fs : int
        The sample rate of the array

    window_size : int
        The size of the window to use for the waveforms

    ma_window : float
        The size of the window to use for the moving average

    threshold_multiplier : float
        The standard deviation multiplier to use for the threshold

    Returns
    ------------
    np.ndarray
        The extracted waveforms
    '''
    spike_times = detect_spikes(arr, fs, ma_window, threshold_multiplier)
    waveforms = [arr[spike - window_size // 2 : 1 + spike + window_size // 2] for spike in spike_times]
    return np.vstack(waveforms)


def compare_clusters(group_1: SortedSpikes, group_2: SortedSpikes) -> tuple[np.ndarray, np.ndarray]:
    '''
    maps the waveforms from group_2 into the PCA space of group_1

    Parameters
    ------------
    group_1 : SortedSpikes
        the group of spikes we will use to define our PCA space, output of ntk.sort_spikes
    
    group_2 : SortedSpikes
        the group of spikes we want to map into group_1s PCA space
    '''

    pca = PCA().fit(group_1._waveforms)
    group_1_transformed = pca.transform(group_1._waveforms)
    group_2_transformed = pca.transform(group_2._waveforms)

    group_1_labels = group_1.labels
    group_2_labels = group_2.labels

    fig = plt.figure(figsize=(8, 8))
    counter = 0
    for idx, cluster in enumerate(np.unique(group_1_labels)):
        mask = group_1_labels == cluster
        plt.scatter(
            group_1_transformed[mask, 0],
            group_1_transformed[mask, 1],
            c=f'C{counter}',
            edgecolors='black',
            linewidths=0.5,
            label=f'Group 1 cluster {cluster}'
        )
        counter += 1
    
    for idx, cluster in enumerate(np.unique(group_2_labels)):
        mask = group_2_labels == cluster
        plt.scatter(
            group_2_transformed[mask, 0],
            group_2_transformed[mask, 1],
            c=f'C{counter}',
            edgecolors='black',
            linewidths=0.5,
            marker='X',
            label=f'Group 2 cluster {cluster}'
        )
        counter += 1
    plt.xlabel('Group 1 PC1')
    plt.ylabel('Group 2 PC2')
    plt.legend()
    plt.show()
        





def sort_spikes(
        arr: np.ndarray,
        fs: int,
        lowcut: float = 100,
        highcut: float = 9000,
        order: int = 4,
        ma_window: float = 0.025,
        threshold_multiplier: float = 4,
        waveform_window: int = 30,
        cluster_dimensions: int = 3
) -> SortedSpikes:
    '''
    Function for sorting spikes from raw data.

    Parameters
    ----------
    arr : np.ndarray
        The raw data to sort.

    fs : int
        The sampling frequency of the raw data.

    lowcut : float, optional
        The lowcut frequency for the bandpass filter. Default is 100.

    highcut : float, optional
        The highcut frequency for the bandpass filter. Default is 9000.

    order : int, optional
        The order of the bandpass filter. Default is 4.

    ma_window : float, optional
        The window size for the moving average filter. Default is 0.025.

    threshold_multiplier : float, optional
        The multiplier for the threshold used to detect spikes. Default is 4.

    waveform_window : int, optional
        The window size for extracting waveforms. Default is 30.

    Returns
    -------
    SortedSpikes
        An object containing the spike sorting results.
    '''

    def _apply_band_pass(arr: np.ndarray) -> np.ndarray:
        '''
        Private method for applying the bandpass filter.
        '''
        filter = Filter(fs=fs, lowcut=lowcut, highcut=highcut, order=order)
        return filter.apply(arr)
    
    
    def _moving_average(arr: np.ndarray) -> np.ndarray:
        '''
        Private method for applying the moving average filter.
        '''
        n = int(fs * ma_window)
        window = np.ones(n) / n
        return convolve(arr, window, mode='same')
    
    def _moving_std(arr: np.ndarray) -> np.ndarray:
        '''
        Private method for applying the moving standard deviation filter.
        '''
        ma = _moving_average(arr)
        ma_sq = _moving_average(arr ** 2)
        return np.sqrt(ma_sq - ma ** 2)
    
    def _detect_spikes(arr: np.ndarray) -> np.ndarray:
        '''
        Private method for detecting spikes.
        '''
        ma = _moving_average(arr)
        mstd = _moving_std(arr)
        threshold = ma + mstd * threshold_multiplier
        peaks, _ = find_peaks(arr, height=threshold)
        return peaks
    
    def _extract_waveforms(arr: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
        '''
        Private method for extracting waveforms.
        '''
        waveforms = [arr[spike - waveform_window:spike + waveform_window] for spike in spike_times]
        return np.vstack(waveforms)
    
    def _apply_pca(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Private method for applying PCA to the waveforms.
        '''
        pca = PCA()
        scaler = StandardScaler(with_std=False)
        scaled_waveforms = scaler.fit_transform(waveforms)
        pca_embeddings = pca.fit_transform(scaled_waveforms)
        return pca_embeddings, pca.explained_variance_ratio_
    
    
    def _fit_clusters(pca_embeddings: np.ndarray) -> np.ndarray:
        '''
        Private method for fitting the clusters. Uses the silhouette score to determine the best number of clusters.
        '''
        
        embeddings = pca_embeddings[:, :cluster_dimensions]

        scores = []
        cluster_range = range(2, 11)
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters).fit(embeddings)
            labels = kmeans.labels_
            scores.append(silhouette_score(embeddings, labels))
        
        best_cluster_number = cluster_range[np.argmax(scores)]
        kmeans = KMeans(best_cluster_number).fit(embeddings)
        
        return kmeans.labels_, scores
    

    def _organize_sort_summary(
            spike_times: np.ndarray,
            waveforms: np.ndarray,
            labels: np.ndarray,
            clustering_scores: np.ndarray,
            pca_embeddings: np.ndarray,
            pca_var_explained: np.ndarray
    ) -> dict:
        
        sort_summary = {}

        sort_summary['clusters'] = {
            cluster : {
                'spike_times' : spike_times[labels == cluster],
                'waveforms' : waveforms[labels == cluster]
            }
            for cluster in np.unique(labels)
        }

        sort_summary['parameters'] = {
            'fs' : fs,
            'lowcut' : lowcut,
            'highcut' : highcut,
            'order' : order,
            'ma_window' : ma_window,
            'threshold_multiplier' : threshold_multiplier,
            'waveform_window' : waveform_window,
            'cluster_dimensions' : cluster_dimensions
        }

        sort_summary['cluster_labels'] = labels
        sort_summary['clustering_scores'] = clustering_scores
        sort_summary['pca_embeddings'] = pca_embeddings
        sort_summary['pca_var_explained'] = pca_var_explained
        sort_summary['raw_data'] = arr
        sort_summary['waveforms'] = waveforms
        sort_summary['spike_times'] = spike_times

        return sort_summary
    

    # Apply bandpass filter
    bandpassed = _apply_band_pass(arr)
    spike_times = _detect_spikes(bandpassed)
    waveforms = _extract_waveforms(arr, spike_times)
    pca_embeddings, pca_var_explained = _apply_pca(waveforms)
    labels, clustering_scores = _fit_clusters(pca_embeddings)
    sort_summary = _organize_sort_summary(spike_times, waveforms, labels, clustering_scores, pca_embeddings, pca_var_explained)
    
    return SortedSpikes(sort_summary)


def area_under_curve(arr: np.ndarray, fs: int, above_baseline: bool) -> float:
    '''
    Compute the area under the curve of a waveform

    Parameters
    ------------
    arr : np.ndarray
        Array to compute area under curve for

    fs : int
        Sample rate of arr
    '''
    if above_baseline:
        arr = arr[arr > 0]
        return np.trapz(arr, dx=1/fs)
    else:
        arr = arr[arr < 0]
        return np.trapz(-arr, dx=1/fs)


def peak_to_peak_amplitude(arr: np.ndarray) -> float:
    '''
    Compute the peak to peak amplitude of a waveform

    Parameters
    ------------

    arr : np.ndarray
        Array to compute peak to peak for
    '''
    return arr.max() - arr.min()


def peak_to_peak_time(arr: np.ndarray, fs: int) -> float:
    '''
    Compute the peak to peak time of a waveform

    Parameters
    ------------

    arr : np.ndarray
        Array to compute peak to peak time for

    fs : int
        Sample rate of arr
    '''
    peak_idx = np.argmax(arr)
    trough_idx = np.argmin(arr)

    return (trough_idx - peak_idx) / fs


def rise_time(arr: np.ndarray, fs: int, threshold_fraction: float) -> float:
    '''
    Compute the rise time of a waveform

    Parameters
    ------------

    arr : np.ndarray
        Array to compute rise time for

    fs : int
        Sample rate of arr

    threshold_fraction: float
        Fraction of the peak amplitude to use as the threshold
    '''
    peak_value = arr.min()
    threshold = peak_value * threshold_fraction
    
    rise_idx = np.where(arr < threshold)[0][0]
    peak_idx = np.argmin(arr)

    return (peak_idx - rise_idx) / fs


def compute_width_at_half_max(arr: np.ndarray, fs: int) -> float:
    '''
    Compute the width at half max of a waveform

    Parameters
    ------------

    arr : np.ndarray
        Array to compute width at half max for

    fs : int
        Sample rate of arr
    '''
    half_max = arr.min() / 2
    half_max_idx = np.where(arr > half_max)[0]

    return (half_max_idx[-1] - half_max_idx[0]) / fs
