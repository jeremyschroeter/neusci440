'''
Neuroscience toolkit
Written for Python 3.12.6
@ Jeremy Schroeter, April 2025
'''

import os
# This line necesarry before numpy to avoid error when running kmeans
os.environ['OMP_NUM_THREADS'] = '1'
import errno

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import find_peaks, butter, filtfilt, fftconvolve

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
        self.n_blocks = len(self.matlab_dict['datastart'][0]) # each block has the same number of channels
        
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
            
            block_data = []
            for channel in range(self.n_channels):
                block_data.append(self.data[f'Channel {channel + 1}'][indices])
            if self.n_channels == 1:
                return np.array(block_data)[0]
            return np.array(block_data)


        # If multiple blocks are requested, return a dictionary of blocks
        data_to_fetch = {}
        for block in indices:
            block_data = []
            for channel in range(self.n_channels):
                block_data.append(self.data[f'Channel {channel + 1}'][block])
            
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
    def comments(self) -> dict:
        '''
        Returns and formats comment data
        '''
        comment_data = np.array(self.matlab_dict['com'], dtype=int)
        comment_text = self.matlab_dict['comtext']

        block_keys = np.unique(comment_data[:, 1]) - 1
        vals = [[[] for channel in range(self.n_channels)] for block in range(self.n_blocks)] ## array of empty lists shape = (nblocks, n channels)
        for comment in comment_data:
            block_num = comment[1]-1 ##adicht uses one based indexing, ntk uses zero based indexing
            channel_num = comment[0]-1
            pos_text = (comment[2], comment_text[comment[4]-1]) ## (sample number, comment text)
            vals[block_num][channel_num].append(pos_text)

        comment_dict = {f'block_{block}' : comments for block, comments in zip(block_keys, vals)} # should keep block nomenclature consistent
        return comment_dict
        
    @property
    def fs(self) -> float | np.ndarray:
        '''
        Returns the sampling frequency of the data. If sampling frequency is constant, returns a float.
        '''
        fs = self.matlab_dict['samplerate']

        if np.all(fs == fs[0]):
            return fs.reshape(-1)[0]
        else:
            return fs


class SortedSpikes:
    '''
    Object for interacting with spike sorting results.

    Parameters
    ----------
    sort_summary : dict
        Output of sort_spikes function.
    '''

    def __init__(
            self,
            sorted_spikes: dict
    ):
        
        # Parse spike sorting results
        self.sorted_spikes = sorted_spikes
        
    def get_cluster_waveforms(
            self,
            cluster: int
    ) -> np.ndarray:
        return self.sorted_spikes[cluster]['waveforms']
    
    def get_cluster_spike_times(
            self,
            cluster: int
    ) -> np.ndarray:
        return self.sorted_spikes[cluster]['spike_times']
    
    def plot_spikes(self) -> None:
        for cluster, cluster_info in self.sorted_spikes.items():
            plt.plot(cluster_info['waveforms'].T, alpha=0.5, c=f'C{cluster}')
            plt.plot(cluster_info['waveforms'].mean(0), c='black')
        plt.show()

    def save(self, file_path: str, compressed: bool = True) -> None:
        '''
        Saves the sorted spike data to a .npz file.

        Parameters
        ----------
        file_path : str
            The file path to save the data (should end with .npz).
        
        compressed : bool
            Whether to use compressed saving with np.savez_compressed. Default = True
        '''
        if not file_path.endswith('.npz'):
            raise ValueError("file_path must end with '.npz'")

        arrays_to_save = {}
        for cluster, cluster_data in self.sorted_spikes.items():
            arrays_to_save[f'cluster_{cluster}_waveforms'] = cluster_data['waveforms']
            arrays_to_save[f'cluster_{cluster}_spike_times'] = cluster_data['spike_times']

        save_func = np.savez_compressed if compressed else np.savez
        save_func(file_path, **arrays_to_save)
    
    @classmethod
    def load(cls, file_path: str) -> "SortedSpikes":
        '''
        Loads sorted spike data from a .npz file and returns a SortedSpikes object.

        Parameters
        ----------
        file_path : str
            Path to the .npz file containing saved spike data.

        Returns
        -------
        SortedSpikes
            Reconstructed SortedSpikes object.
        '''
        data = np.load(file_path)
        clusters = {}

        for key in data.files:
            if key.endswith('waveforms'):
                cluster = int(key.split('_')[1])
                if cluster not in clusters:
                    clusters[cluster] = {}
                clusters[cluster]['waveforms'] = data[key]
            elif key.endswith('spike_times'):
                cluster = int(key.split('_')[1])
                if cluster not in clusters:
                    clusters[cluster] = {}
                clusters[cluster]['spike_times'] = data[key]

        return cls(clusters)


def bandpass_filter(
        signal: np.ndarray,
        fs: float,
        lowcut: float = 300,
        highcut: float = 3000
) -> np.ndarray:
    '''
    Applies a bandpass filter to the signal
    '''
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_spikes(
        signal: np.ndarray,
        threshold: np.ndarray,
        fs: int,
        window_ms = 2
) -> tuple[np.ndarray]:
    '''
    Detects AP times and returns their waveforms and spike times

    Parameters
    ----------
    signal : np.ndarray
        1D numpy array containing the signal to be spike sorted
    
    threshold : float
        threshold above which to detect peaks
    
    fs : int
        sample rate
    
    window_ms : float
        length of waveform window in milliseconds
    
    Returns
    ----------
    waveforms : np.ndarray
        waveforms of detected actions potentials as a (N, T) array
    
    spike_times : np.ndarray
        detected AP times as a (N,) array
    '''

    # Get maximum across times where signal crosses threshold
    # We use the inverted signal here because find_peaks
    # looks for minima not maxima, extracellular spikes are
    # downward reflecting (mostly...)
    if signal.ndim != 1:
        raise ValueError('signal is not 1-dimensional')
    peaks, _ = find_peaks(-signal, height=threshold, distance=int(0.001 * fs))

    # Extract waveforms
    half_window = int((fs / 1000) * (window_ms / 2))
    waveforms = []
    valid_spike_times = []
    for spike_time in peaks:
        if spike_time - half_window > 0 and spike_time + half_window < len(signal):
            window = signal[spike_time - half_window : spike_time + half_window]
            waveforms.append(window)
            valid_spike_times.append(spike_time)
            
    return np.stack(waveforms), np.array(valid_spike_times)

def approx_firing_rate(
        signal: np.ndarray,
        threshold: np.ndarray,
        fs: int,
        sigma: float = 0.1

) -> np.ndarray:
    '''
    Computes an approximate firing rate

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal

    threshold : float
        threshold above which to detect spikes
    
    fs : int
        sampling rate of the signal

    sigma : float
        temporal resolution of kernel

    Returns
    ----------
    np.ndarray
        array of the approximate firing rate at each time in the signal

    '''

    g_kernel = lambda x: np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)) # smooth with a gaussian kernel
    x = np.linspace(-10, 10, int(20*fs)) 

    times, _ = find_peaks(-signal, height=threshold, distance=int(0.001 * fs))
    dense_times = np.zeros(len(signal))
    for time in times:
        dense_times[time] = 1

    smoothed_fr = fftconvolve(dense_times, g_kernel(x), mode='same') # units work out to sp/sec b/c integral=1

    return smoothed_fr


def apply_PCA(
        X: np.ndarray,
        dims_to_keep: int = 2
) -> np.ndarray:
    '''
    Project array onto its principal components

    Parameters
    ----------
    X : np.ndarray
        Data matrix to perform PCA on. Should have shape (N, D) where
        N is the number of data points and D is the number of features
    
    dims_to_keep : int
        Output dimensionality of the projected data
    
    Returns
    ----------
    np.ndarray
        Projected data
    '''
    pca = PCA(n_components=dims_to_keep)
    return pca.fit_transform(X)


def cluster_waveforms(
        waveforms: np.ndarray,
        n_clusters: int
) -> np.ndarray:
    """
    Cluster waveforms using the K-Means algorithm.
    Parameters:
        waveforms (np.ndarray): A 2D array where each row represents a waveform to be clustered.
        n_clusters (int): The number of clusters to form.
    Returns:
        np.ndarray: A 1D array of cluster labels corresponding to each waveform.
    """
    kmeans = KMeans(n_clusters)
    return kmeans.fit_predict(waveforms)


def sort_spikes(
        signal: np.ndarray,
        threshold: np.ndarray,
        fs: int,
        n_clusters: int,
        pca_dims: int = 2
) -> SortedSpikes:
    '''
    Sorts spikes for a 1D extracellular recording.

    Parameters
    ----------
    signal : np.ndarray
        The 1D recording to spike sort, expected to have shape (T,)
    
    threshold : np.ndarray
        Threshold to use for detecting spikes (if you don't know what to put
        try doing k * np.median(np.abs(signal)) and playing with k)
    
    fs : int
        Sample rate used in the recording
    
    n_clusters : int
        Number of clusters for KMeans to run, should be equal to the number
        of neurons you think are present in the signal
    
    pca_dims : int
        Number of principal components to use when clustering the waveforms.
        Default = 2
    '''
    
    filtered_signal = bandpass_filter(signal, fs)
    waveforms, spike_times = detect_spikes(filtered_signal, threshold, fs)
    pca_waveforms = apply_PCA(waveforms, pca_dims)
    labels = cluster_waveforms(pca_waveforms, n_clusters)

    sort_summary = {
        cluster : {
            'waveforms' : waveforms[labels == cluster],
            'spike_times' : spike_times[labels ==  cluster]
        }
        for cluster in np.unique(labels)
    }

    return SortedSpikes(sort_summary)