'''
Neuroscience toolkit
Written for Python 3.12.4
@ Jeremy Schroeter, August 2024
'''

import os
import errno
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt
from subprocess import PIPE, run
from scipy.io import loadmat
from scipy import signal
from scipy.signal import lfilter, butter, filtfilt, dimpulse, find_peaks, freqz
from sklearn.cluster import KMeans
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


