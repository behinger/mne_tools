# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:48:20 2016
based on mne.io.read_raw_eeglab
@author: behinger
"""


import os.path as op

import numpy as np

#from ..utils import (_read_segments_file, _find_channels,_synthesize_stim_channel)
from mne.io.utils import (_read_segments_file, _find_channels,_synthesize_stim_channel)
#from ..constants import FIFF
#from ..meas_info import _empty_info, create_info
from mne.io.constants import FIFF
from mne.io.meas_info import _empty_info,create_info
#from ..base import _BaseRaw, _check_update_montage
from mne.io.base import BaseRaw
#from ...utils import logger, verbose, check_version, warn
from mne.utils import logger, verbose, warn
#from ...channels.montage import Montage
#from ...epochs import _BaseEpochs
#from ...event import read_events
#from ...externals.six import string_types
from mne.externals.six import string_types

CAL = 1e-6

def _check_fname(fname):
    """Check if the file extension is valid."""
    fmt = str(op.splitext(fname)[-1])
    if fmt == '.cnt':
        raise IOError('Expected .cnt file format. Found %s format' % fmt)




def _to_loc(ll):
    """Check if location exists."""
    if isinstance(ll, (int, float)) or len(ll) > 0:
        return ll
    else:
        return 0.


def _get_info(eeg, montage, eog=()):
    """Get measurement info."""
    info = _empty_info(sfreq=eeg.get_sample_frequency())

    # add the ch_names and info['chs'][idx]['loc']
    
    ch_names =[eeg.get_channel(i)[0] for i in range(eeg.get_channel_count())]
    print ch_names
        
    #elif isinstance(montage, string_types):
    #    path = op.dirname(montage)
    #else:  # if eeg.chanlocs is empty, we still need default chan names
    #    ch_names = ["EEG %03d" % ii for ii in range(eeg.get_channel_count())]

    #if montage is None:
    info = create_info(ch_names, eeg.get_sample_frequency(), ch_types='eeg')
    #else:
    #    _check_update_montage(info, montage, path=path,
    #                          update_ch_names=True)

    info['buffer_size_sec'] = 1.  # reasonable default
    # update the info dict

    if eog == 'auto':
        eog = _find_channels(ch_names)

    for idx, ch in enumerate(info['chs']):
        ch['cal'] = CAL
        if ch['ch_name'] in eog or idx in eog:
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE
            ch['kind'] = FIFF.FIFFV_EOG_CH
    return info


def read_raw_antcnt(input_fname, montage=None, eog=(), event_id=None,
                    event_id_func='strip_to_integer', preload=False,
                    verbose=None):
    """Read an ANT .cnt file
    Parameters
    ----------
    input_fname : str
        Path to the .cnt file. 
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    event_id : dict | None
        The ids of the events to consider. If None (default), an empty dict is
        used and ``event_id_func`` (see below) is called on every event value.
        If dict, the keys will be mapped to trigger values on the stimulus
        channel and only keys not in ``event_id`` will be handled by
        ``event_id_func``. Keys are case-sensitive.
        Example::
            {'SyncStatus': 1; 'Pulse Artifact': 3}
    event_id_func : None | str | callable
        What to do for events not found in ``event_id``. Must take one ``str``
        argument and return an ``int``. If string, must be 'strip-to-integer',
        in which case it defaults to stripping event codes such as "D128" or
        "S  1" of their non-integer parts and returns the integer.
        If the event is not in the ``event_id`` and calling ``event_id_func``
        on it results in a ``TypeError`` (e.g. if ``event_id_func`` is
        ``None``) or a ``ValueError``, the event is dropped.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory). Note that
        preload=False will be effective only if the data is stored in a
        separate binary file.
    verbose : bool | str | int | None
        If not None, override default verbose level (see mne.verbose).
  
    Returns
    -------
    raw : Instance of RawANTCNT
        A Raw object containing ANT .cnt data.
    Notes
    -----
    .. versionadded:: 0.11.0
    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawANTCNT(input_fname=input_fname, montage=montage, preload=preload,
                     eog=eog, event_id=event_id, event_id_func=event_id_func,
                     verbose=verbose)





class RawANTCNT(BaseRaw):
    """Raw object from ANT .cnt file.
    Parameters
    ----------
    input_fname : str
        Path to the .cnt file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .cnt file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions. If None,
        sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    event_id : dict | None
        The ids of the events to consider. If None (default), an empty dict is
        used and ``event_id_func`` (see below) is called on every event value.
        If dict, the keys will be mapped to trigger values on the stimulus
        channel and only keys not in ``event_id`` will be handled by
        ``event_id_func``. Keys are case-sensitive.
        Example::
            {'SyncStatus': 1; 'Pulse Artifact': 3}
    event_id_func : None | str | callable
        What to do for events not found in ``event_id``. Must take one ``str``
        argument and return an ``int``. If string, must be 'strip-to-integer',
        in which case it defaults to stripping event codes such as "D128" or
        "S  1" of their non-integer parts and returns the integer.
        If the event is not in the ``event_id`` and calling ``event_id_func``
        on it results in a ``TypeError`` (e.g. if ``event_id_func`` is
        ``None``) or a ``ValueError``, the event is dropped.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires large
        amount of memory). If preload is a string, preload is the file name of
        a memory-mapped file which is used to store the data on the hard
        drive (slower, requires less memory).
    verbose : bool | str | int | None
        If not None, override default verbose level (see mne.verbose).
 
    Returns
    -------
    raw : Instance of RawANTCNT
        A Raw object containing ANT .cnt data.
    Notes
    -----
    .. versionadded:: 0.11.0
    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, montage, eog=(), event_id=None,
                 event_id_func='strip_to_integer', preload=False,
                 verbose=None):
        """Read ANT .cnt file.
        """
        #from scipy import io
        import libeep
        #basedir = op.dirname(input_fname)
        eeg= libeep.read_cnt(input_fname)
        
        last_samps = [eeg.get_sample_count() - 1]
        info = _get_info(eeg, montage, eog=eog)

        stim_chan = dict(ch_name='STI 014', coil_type=FIFF.FIFFV_COIL_NONE,
                         kind=FIFF.FIFFV_STIM_CH, logno=len(info["chs"]) + 1,
                         scanno=len(info["chs"]) + 1, cal=1., range=1.,
                         loc=np.zeros(12), unit=FIFF.FIFF_UNIT_NONE,
                         unit_mul=0., coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
        info['chs'].append(stim_chan)
        info._update_redundant()

        events = _read_antcnt_events(eeg, event_id=event_id,
                                     event_id_func=event_id_func)
        self._create_event_ch(events, n_samples=eeg.get_sample_count())

        # read the data
    
        if preload is False or isinstance(preload, string_types):
            warn('Data will be preloaded. preload=False or a string '
                 'preload is not supported when the data is stored in '
                 'the .cnt file')
        # don't know how to implement preload = F
        
        n_chan = eeg.get_channel_count()
        n_times = eeg.get_sample_count()
        
        data = np.empty((n_chan+1, n_times), dtype=np.double)
        from numpy import asarray
        x = asarray(eeg.get_samples(0,n_times))
        x.shape  = (n_times,n_chan) 
        
        data[:-1] = x.transpose()
        
        
        data *= CAL
        data[-1] = self._event_ch
        super(RawANTCNT, self).__init__(
            info, data, last_samps=last_samps, orig_format='double',
            verbose=verbose)

    def _create_event_ch(self, events, n_samples=None):
        """Create the event channel"""
        if n_samples is None:
            n_samples = self.last_samp - self.first_samp + 1
        events = np.array(events, int)
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError("[n_events x 3] shaped array required")
        # update events
        self._event_ch = _synthesize_stim_channel(events, n_samples)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                            dtype=np.float32, trigger_ch=self._event_ch,
                            n_channels=self.info['nchan'] - 1)


def _read_antcnt_events(eeg, event_id=None, event_id_func='strip_to_integer'):
        """Create events array from ANT cnt structure
        An event array is constructed by looking up events in the
        event_id, trying to reduce them to their integer part otherwise, and
        entirely dropping them (with a warning) if this is impossible.
        Returns a 1x3 array of zeros if no events are found."""
        if event_id_func is 'strip_to_integer':
            event_id_func = _strip_to_integer
        if event_id is None:
            event_id = dict()
    
        types = [eeg.get_trigger(i)[0] for i in range(eeg.get_trigger_count())]
        latencies= [eeg.get_trigger(i)[1] for i in range(eeg.get_trigger_count())]
            
            
        
        if len(types) < 1:  # if there are 0 events, we can exit here
            logger.info('No events found, returning empty stim channel ...')
            return np.zeros((0, 3))
    
        not_in_event_id = set(x for x in types if x not in event_id)
        not_purely_numeric = set(x for x in not_in_event_id if not x.isdigit())
        no_numbers = set([x for x in not_purely_numeric
                          if not any([d.isdigit() for d in x])])
        have_integers = set([x for x in not_purely_numeric
                             if x not in no_numbers])
        if len(not_purely_numeric) > 0:
            basewarn = "Events like the following will be dropped"
            n_no_numbers, n_have_integers = len(no_numbers), len(have_integers)
            if n_no_numbers > 0:
                no_num_warm = " entirely: {0}, {1} in total"
                warn(basewarn + no_num_warm.format(list(no_numbers)[:5],
                                                   n_no_numbers))
            if n_have_integers > 0 and event_id_func is None:
                intwarn = (", but could be reduced to their integer part "
                           "instead with the default `event_id_func`: "
                           "{0}, {1} in total")
                warn(basewarn + intwarn.format(list(have_integers)[:5],
                                               n_have_integers))
    
        events = list()
        for tt, latency in zip(types, latencies):
            try:  # look up the event in event_id and if not, try event_id_func
                event_code = event_id[tt] if tt in event_id else event_id_func(tt)
                events.append([int(latency), 1, event_code])
            except (ValueError, TypeError):  # if event_id_func fails
                pass  # We're already raising warnings above, so we just drop
    
        if len(events) < len(types):
            missings = len(types) - len(events)
            msg = ("{0}/{1} event codes could not be mapped to integers. Use "
                   "the 'event_id' parameter to map such events manually.")
            warn(msg.format(missings, len(types)))
            if len(events) < 1:
                warn("As is, the trigger channel will consist entirely of zeros.")
                return np.zeros((0, 3))
    
        return np.asarray(events)
    
    
def _strip_to_integer(trigger):
    """Return only the integer part of a string."""
    return int("".join([x for x in trigger if x.isdigit()]))

