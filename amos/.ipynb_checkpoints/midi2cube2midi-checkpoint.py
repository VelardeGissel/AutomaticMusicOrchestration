#GV 25.Sep-2025
import sys
sys.path.append('amos') 
from amos.midi2df2midi import midi_to_dataframe, save_midi_from_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Union

#GV. 29.9.2025
def midi_to_cubematrix(filein,samples_per_qn=12):
    dfnmat = midi_to_dataframe(filein)
    pianoroll, tc_to_index, index_to_tc, tcp_to_trackname = df2cubematrix(dfnmat, samples_per_qn=samples_per_qn)
    return pianoroll, tc_to_index, index_to_tc, tcp_to_trackname

"""
def cubematrix_to_midi(fileout,pianoroll,samples_per_qn=12, tcp_to_index= tcp_to_index, index_to_tcp = index_to_tcp, tcp_to_trackname=tcp_to_trackname, min_activation = min_activation):
    # Invert the cube back to a note list
    dfnmat = cubematrix2df(pianoroll, samples_per_qn=12, 
    tcp_to_index = tcp_to_index,
    index_to_tcp = index_to_tcp,
    tcp_to_trackname = tcp_to_trackname,
    min_activation = min_activation)
"""
def save_midi(dfnmat, fileout):
    # Convert to DataFrame
    dfdata = pd.DataFrame(dfnmat, columns=[
            'track number','track name', 'channel', 'program',
            'onset in quarter notes','duration in quarter notes','pitch','velocity'
        ])

    # Fix data types
    dfdata['track number'] = dfdata['track number'].astype(int)
    dfdata['channel'] = dfdata['channel'].astype(int)
    dfdata['program'] = dfdata['program'].astype(int)
    dfdata['pitch'] = dfdata['pitch'].astype(int)
    dfdata['velocity'] = dfdata['velocity'].astype(int)
    dfdata['onset in quarter notes'] = dfdata['onset in quarter notes'].astype(float)
    dfdata['duration in quarter notes'] = dfdata['duration in quarter notes'].astype(float)
    dfdata['track name'] = dfdata['track name'].astype(str)
    save_midi_from_df(dfdata, fileout)


"""
GV. 26.09.2025
GPT-5 text.format: text
effort: medium
verbosity: medium
summary: auto
store: true
Prompt:
create or update the python function called df2cubematrix that takes as input the dfnmat and
returns a 3-D pianoroll matrix of pitch, time, and track-channel-program information.
dfnmat comes from a midi file and has information in 8 columns for each note: track number (int),track name (string), channel (int), 
program (int), onset in quarter notes (float), duration in quarter notes (float),
pitch (int), velocity (int in range 0 and 127).
originally, pianoroll was a matrix of zeros and ones
of size: 128 (rows), 12*duration in quarter notes (columns), 
and m matrices for each track-channel-program as depth.
Now, it should be a matrix normalized between zero and one, where the activated values 
correspond to the normalized velocity, original between 0 and 127, normalized between 0 and 1. 

Rows correspond to 0 to 128 pitch numbers. Columns correspond to a time grid of duration in quarter notes, sampled in 12 samples per quarter note.
The depth corresponds m track-channels-program.

There should be m unique track-channel-program combinations. Therefore, each pitch-time matrix corresponds to the piano roll representation for each track-channel-program combination.

Create the inverse function called cubematrix2df, such that it takes  3-D pianoroll (pitch x time x track-channel-program) back into a note list (dfnmat).

Create a function to plot a given pianoroll part, when provided the piano roll matrix, start and end time in quarter notes, and track-channel combination.

"""
#import numpy as np
#import matplotlib.pyplot as plt
#from collections import OrderedDict
#from typing import Dict, List, Tuple, Optional, Union


# ---------- Helpers ----------

def _as_numpy(a) -> np.ndarray:
    """Accept numpy array or pandas DataFrame; return numpy 2D array."""
    try:
        return a.to_numpy()
    except AttributeError:
        return np.asarray(a)


def _frame_bounds(onset_qn: float, dur_qn: float, spq: int, T: int) -> Tuple[int, int]:
    """Convert onset/duration in quarter notes to [start,end) frame indices on a grid."""
    start = int(np.floor(onset_qn * spq))
    end = int(np.ceil((onset_qn + max(0.0, dur_qn)) * spq))
    if dur_qn > 0 and end <= start:
        end = start + 1
    start = max(0, min(start, max(T - 1, 0)))
    end = max(start + 1, min(end, T))
    return start, end


# ---------- Main functions ----------

def df2cubematrix(
    dfnmat: Union[np.ndarray, "pandas.DataFrame"],
    samples_per_qn: int = 12
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int], List[Tuple[int, int, int]], Dict[Tuple[int, int, int], str]]:
    """
    Convert a note matrix (dfnmat) into a 3-D piano-roll tensor with normalized velocities.

    Input (dfnmat): rows = notes, columns =
      0: track number (int)
      1: track name (string)
      2: channel (int)
      3: program (int)
      4: onset in quarter notes (float)
      5: duration in quarter notes (float)
      6: pitch (int, 0..127)
      7: velocity (int, 0..127)

    Returns:
      pianoroll: np.ndarray of shape (128, T, m) with values in [0,1] (dtype float32)
                 where T = ceil(total_duration_qn * samples_per_qn)
                       m = number of unique (track, channel, program) combos
      tcp_to_index: dict mapping (track, channel, program) -> depth index in pianoroll
      index_to_tcp: list of (track, channel, program) in depth order
      tcp_to_trackname: dict mapping (track, channel, program) -> track name (string)

    Notes:
      - Time grid uses samples_per_qn samples per quarter note (default: 12).
      - When multiple notes overlap at the same (pitch, time, tcp), the max normalized
        velocity is used (i.e., union via max).
      - Velocity is normalized from [0,127] to [0,1] as vel_norm = velocity/127.
    """
    arr = _as_numpy(dfnmat)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError("dfnmat must be a 2-D array-like with 8 columns.")

    # Build mapping (track, channel, program) -> depth index and track names
    tcp_to_index: "OrderedDict[Tuple[int,int,int], int]" = OrderedDict()
    tcp_to_trackname: Dict[Tuple[int, int, int], str] = {}
    total_end_idx = 0

    # First pass: determine T and mappings
    for row in arr:
        track = int(row[0])
        tname = str(row[1])
        channel = int(row[2])
        program = int(row[3])
        onset_qn = float(row[4])
        dur_qn = float(row[5])
        pitch = int(row[6])

        if not (0 <= pitch <= 127):
            continue
        if dur_qn < 0:
            continue

        tcp = (track, channel, program)
        if tcp not in tcp_to_index:
            tcp_to_index[tcp] = len(tcp_to_index)
            tcp_to_trackname[tcp] = tname  # first-seen name for this tcp

        # update T bound
        end_idx = int(np.ceil((onset_qn + dur_qn) * samples_per_qn))
        total_end_idx = max(total_end_idx, end_idx)

    m = len(tcp_to_index)
    T = max(total_end_idx, 1)

    # Allocate cube: pitches x time x (track,channel,program)
    pianoroll = np.zeros((128, T, m), dtype=np.float32)

    # Second pass: fill with normalized velocity values via max
    for row in arr:
        track = int(row[0])
        channel = int(row[2])
        program = int(row[3])
        onset_qn = float(row[4])
        dur_qn = float(row[5])
        pitch = int(row[6])
        velocity = int(row[7])

        if not (0 <= pitch <= 127) or dur_qn < 0:
            continue
        tcp = (track, channel, program)
        depth = tcp_to_index[tcp]

        start_idx, end_idx = _frame_bounds(onset_qn, dur_qn, samples_per_qn, T)
        vel_norm = float(np.clip(velocity, 0, 127)) / 127.0
        # Merge by taking max across overlaps
        pianoroll[pitch, start_idx:end_idx, depth] = np.maximum(
            pianoroll[pitch, start_idx:end_idx, depth],
            vel_norm
        )

    index_to_tcp = [None] * m
    for tcp, idx in tcp_to_index.items():
        index_to_tcp[idx] = tcp

    return pianoroll, dict(tcp_to_index), index_to_tcp, tcp_to_trackname

def cubematrix2df(
    pianoroll: np.ndarray,
    samples_per_qn: int = 12,
    tcp_to_index: Optional[Dict[Tuple[int, int, int], int]] = None,
    index_to_tcp: Optional[List[Tuple[int, int, int]]] = None,
    tcp_to_trackname: Optional[Dict[Tuple[int, int, int], str]] = None,
    min_activation: float = 1e-9
) -> np.ndarray:
    """
    Invert a 3-D pianoroll (with normalized velocities) back into a note list (dfnmat).

    Parameters
    - pianoroll: np.ndarray of shape (128, T, m) with values in [0,1] (float)
                 pitch 0..127, time frames, depth m = unique (track, channel, program)
    - samples_per_qn: int, number of frames per quarter note (default 12)
    - tcp_to_index: optional dict mapping (track, channel, program) -> depth index
    - index_to_tcp: optional list of (track, channel, program) ordered by depth
    - tcp_to_trackname: optional dict mapping (track, channel, program) -> track name
    - min_activation: float, values <= min_activation are treated as silence

    Returns
    - dfnmat: numpy array of shape (N_notes, 8) with dtype=object:
        0: track number (int)
        1: track name (string)
        2: channel (int)
        3: program (int)
        4: onset in quarter notes (float)
        5: duration in quarter notes (float)
        6: pitch (int, 0..127)
        7: velocity (int, 0..127; derived from max activation in the run)

    Notes
    - Contiguous runs of activation > min_activation along time for each (pitch, depth)
      are converted to notes.
    - Velocity is recovered by taking the max activation value in the run, un-normalized:
      vel_int = round(clip(max_val, 0, 1) * 127).
    - Exact original note boundaries may not be recoverable if overlapping notes
      of the same pitch in the same TCP merged in the roll (common pianoroll limitation).
    """
    if pianoroll.ndim != 3 or pianoroll.shape[0] != 128:
        raise ValueError("pianoroll must have shape (128, T, m).")
    _, T, m = pianoroll.shape

    # Resolve index_to_tcp (and default track names)
    if index_to_tcp is None:
        if tcp_to_index is not None:
            index_to_tcp = [None] * m
            for tcp, idx in tcp_to_index.items():
                if idx < 0 or idx >= m:
                    raise ValueError(f"tcp_to_index maps to invalid depth {idx} for key {tcp}.")
                index_to_tcp[idx] = tcp
            # Fill any missing with defaults
            for d in range(m):
                if index_to_tcp[d] is None:
                    index_to_tcp[d] = (int(d), 0, 0)
        else:
            index_to_tcp = [(int(d), 0, 0) for d in range(m)]

    # Default names if not given
    def _default_name(tcp: Tuple[int, int, int]) -> str:
        tr, ch, pr = tcp
        return f"track_{tr}_ch_{ch}_pr_{pr}"

    notes: List[List[object]] = []

    for d in range(m):
        track, channel, program = index_to_tcp[d]
        tname = tcp_to_trackname.get((track, channel, program), _default_name((track, channel, program))) if tcp_to_trackname else _default_name((track, channel, program))

        layer = pianoroll[:, :, d]  # (128, T)
        for pitch in range(128):
            row = layer[pitch]  # (T,)
            # Activation mask
            mask = row > float(min_activation)
            if not np.any(mask):
                continue

            # Find runs of True in mask
            padded = np.concatenate(([False], mask, [False]))
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if e <= s:
                    continue
                # Onset/duration
                onset_qn = s / float(samples_per_qn)
                duration_qn = (e - s) / float(samples_per_qn)
                # Velocity from max activation in this segment
                seg_max = float(np.max(row[s:e]))
                vel_int = int(np.clip(np.round(seg_max * 127.0), 0, 127))
                # Enforce a minimum of 1 if it's treated as a note (optional)
                # if vel_int == 0: vel_int = 1

                notes.append([
                    int(track),          # 0 track number
                    tname,               # 1 track name
                    int(channel),        # 2 channel
                    int(program),        # 3 program
                    float(onset_qn),     # 4 onset (qn)
                    float(duration_qn),  # 5 duration (qn)
                    int(pitch),          # 6 pitch
                    int(vel_int)         # 7 velocity
                ])

    if not notes:
        return np.empty((0, 8), dtype=object)

    # Sort for readability
    notes.sort(key=lambda r: (r[4], r[0], r[2], r[3], r[6]))  # onset, then track, ch, prog, pitch
    return np.array(notes, dtype=object)


def plot_pianoroll_part(
    pianoroll: np.ndarray,
    start_qn: float,
    end_qn: float,
    track_channel_program: Union[Tuple[int, int, int], int],
    tcp_to_index: Optional[Dict[Tuple[int, int, int], int]] = None,
    samples_per_qn: int = 12,
    figsize=(10, 4),
    cmap='magma'
):
    """
    Plot a slice of the piano-roll for a given (track, channel, program) and time range.

    Parameters:
      pianoroll: np.ndarray of shape (128, T, m), values in [0,1]
      start_qn: float, start time in quarter notes
      end_qn: float, end time in quarter notes
      track_channel_program: either (track, channel, program) tuple OR an integer depth index
      tcp_to_index: dict mapping (track, channel, program) -> depth index (required if tuple is given)
      samples_per_qn: samples per quarter note (default 12)
      figsize: matplotlib figure size
      cmap: colormap for imshow. cmap='Greys' 

    Behavior:
      - Time axis in quarter notes.
      - Pitches on y-axis (0..127), origin at bottom.
      - Colors represent normalized velocity (0..1).
    """
    if pianoroll.ndim != 3 or pianoroll.shape[0] != 128:
        raise ValueError("pianoroll must have shape (128, T, m).")
    T = pianoroll.shape[1]
    m = pianoroll.shape[2]

    # Resolve depth index
    if isinstance(track_channel_program, tuple):
        if tcp_to_index is None:
            raise ValueError("tcp_to_index must be provided when track_channel_program is a (track, channel, program) tuple.")
        if track_channel_program not in tcp_to_index:
            raise KeyError(f"Track-channel-program {track_channel_program} not found in mapping.")
        depth_idx = tcp_to_index[track_channel_program]
        title_tcp = f"(track={track_channel_program[0]}, channel={track_channel_program[1]}, program={track_channel_program[2]})"
    elif isinstance(track_channel_program, (int, np.integer)):
        depth_idx = int(track_channel_program)
        if depth_idx < 0 or depth_idx >= m:
            raise IndexError(f"Depth index out of range: {depth_idx} (m={m}).")
        title_tcp = f"(depth index={depth_idx})"
    else:
        raise TypeError("track_channel_program must be either (track, channel, program) tuple or an integer depth index.")

    # Compute frame indices
    s = max(0, int(np.floor(start_qn * samples_per_qn)))
    e = min(T, int(np.ceil(end_qn * samples_per_qn)))
    if e <= s:
        e = min(s + 1, T)

    roll_slice = pianoroll[:, s:e, depth_idx]

    plt.figure(figsize=figsize)
    extent = [s / samples_per_qn, e / samples_per_qn, 0, 127]
    im = plt.imshow(roll_slice, origin='lower', aspect='auto', interpolation='nearest',
                    cmap=cmap, extent=extent, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, label='Normalized velocity (0..1)')
    plt.xlabel("Time (quarter notes)")
    plt.ylabel("MIDI pitch")
    plt.title(f"Piano-roll {title_tcp} | {start_qn}–{end_qn} qn")
    plt.tight_layout()
    plt.show()