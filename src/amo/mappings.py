#import numpy as np
from __future__ import annotations

import numpy as np
import mido
from mido import MidiFile, MidiTrack, MetaMessage, Message
"""
gpt-5
text.format: text
effort: medium
verbosity: medium
summary: auto
26.8
Prompt by GV: 
create a python function to return all unique combinations of the first 4 columns of nmat as an array of size N by 4. For example nmat array([[12, 'Violoncello', 0, 45, 0.0, 0.5, 52, 32],

[13, 'Contrabass', 1, 45, 0.0, 0.5, 40, 32],

[9, 'Violin I', 13, 45, 0.5, 0.5, 64, 33],

[10, 'Violin II', 14, 45, 0.5, 0.5, 59, 33],

[11, 'Viola', 15, 45, 0.5, 0.5, 55, 34],

[12, 'Violoncello', 0, 45, 1.0, 0.5, 52, 31],

[13, 'Contrabass', 1, 45, 1.0, 0.5, 40, 30],

[9, 'Violin I', 13, 45, 1.5, 0.5, 66, 32],

[10, 'Violin II', 14, 45, 1.5, 0.5, 60, 30],

[11, 'Viola', 15, 45, 1.5, 0.5, 57, 36]], dtype=object)
"""
def learn_quaterna_mapping(nmat,ytarget="track-channel"):
    """
    Code generated with https://platform.openai.com/
    Model gpt-4.1, text.format: text, temp: 1.00, tokens: 2048, top_p: 1.00, store: true
    Prompt by Gissel Velarde
    Prompt:
    the first 4 columns of a an numpy array with 8 columns always appear in quaterna. Write a function that learns the quaterna, given that column 3 is used as a label for a machine learning model. Then, once the model predicts the label for column 4, fill the corresponding values for columns 0, 1 and 2.
    """
    if ytarget=="program":
        """
        Enhanced to preserve ALL track-channel combinations for each program.
        Returns a dictionary: label_val -> list of [col0, col1, col2]
        """
        mapping = {}
        for row in nmat:
            label = row[3]  # program number
            track_info = row[:3].tolist()  # [track_num, track_name, channel]
            
            if label not in mapping:
                mapping[label] = []
            
            # Only add if this specific combination isn't already present
            if track_info not in mapping[label]:
                mapping[label].append(track_info)
        return mapping
    if ytarget=="instrument-name":
        # Extract the first 4 columns
        first_four = nmat[:, :4]

        # 1. Unique values based ONLY on column 2 (track name)
        col2 = first_four[:, 1]
        unique_col2, indices = np.unique(col2, return_index=True)

        # 2. First column: ascending order (track number)
        col1_new = np.arange(1, len(unique_col2) + 1)

        # 3. Third column: all ones (channel)
        col3_new = np.ones(len(unique_col2), dtype=first_four.dtype)

        # 4. Fourth column: pick original corresponding value
        col4_new = first_four[indices, 3]

        # Build the final mapping array
        mapping = np.column_stack([col1_new, unique_col2, col3_new, col4_new])
    else:
        """
        When target is track-channel
        Returns all unique combinations of the first 4 columns of a NumPy array,
        sorted by the first and third columns.
    
        Args:
            nmat (np.ndarray): A NumPy array with at least 4 columns.
    
        Returns:
            np.ndarray: A new NumPy array containing only the unique rows
                        from the first 4 columns of the input array, sorted.
        """
        # Extract the first 4 columns
        first_four_columns = nmat[:, :4]
    
        # Convert each row (list) into a tuple so it can be added to a set
        unique_rows_as_tuples = set(map(tuple, first_four_columns))
    
        # Convert the set of tuples back to a list of lists and sort it.
        unique_rows = sorted(list(unique_rows_as_tuples), key=lambda x: (x[0], x[2]))
        mapping = np.array(unique_rows, dtype=object)
    # Convert the sorted list back to a NumPy array
    return mapping


#26.8
#Gemini
"""
Prompt GV.
Given y which has the first and third column of map_array, create a function that returns nmat_L, if a combination of first, second, and third column repeats with a different number in the fourth column, insert in nmat_L, only the first combination of the first 3 columns. 

For example:
y =  ['1_1', '1_1', '2_1', '3_1', '4_2', '4_2']    
          
 map_array = np.array([[1, 'V',1,70],
                          [2, 'B',1,72],
                          [3, 'F',1,73],
                          [4, 'F',2,73]]
                          [3, 'F',1,98],
                          [4, 'F',2,98]], dtype=object)
                          
                          
nmat_L = np.array([[1,'V' ,1,70],
                           [1,'V', 1,70],
                           [2, 'B',1,72],
                           [3,'F', 1,73],
                           [4, 'F',2,73],
                           [4,'F', 2,73]], dtype=object)
"""

def fill_quaterna_columns(y, map_array, ytarget="track-channel"):
    if ytarget=="program":
        """
        Code generated with https://platform.openai.com/
        Model gpt-4.1, text.format: text, temp: 1.00, tokens: 2048, top_p: 1.00, store: true
        Prompt by Gissel Velarde
        Prompt:
        the first 4 columns of a an numpy array with 8 columns always appear in quaterna. Write a function that learns the quaterna, given that column 3 is used as a label for a machine learning model. Then, once the model predicts the label for column 4, fill the corresponding values for columns 0, 1 and 2.
        
        Revised by GPT-5 mini.
        """
        """
        Given a list/array of predicted labels, use the mapping to reconstruct cols 0, 1, 2.
        For programs with multiple instruments, distributes notes in round-robin fashion.
        Returns an array of shape (N, 3) where N is the number of predicted labels.
        """
        filled = []
        # Keep track of which instrument to use next for each program (round-robin)
        instrument_counters = {}
        
        for label in y:
            if label in map_array:
                available_instruments = map_array[label]
                
                if len(available_instruments) == 1:
                    # Single instrument - use it directly
                    filled.append(available_instruments[0])
                else:
                    # Multiple instruments - distribute in round-robin fashion
                    if label not in instrument_counters:
                        instrument_counters[label] = 0
                    
                    # Select the next instrument in rotation
                    selected_instrument = available_instruments[instrument_counters[label]]
                    filled.append(selected_instrument)
                    
                    # Move to next instrument for this program
                    instrument_counters[label] = (instrument_counters[label] + 1) % len(available_instruments)
            else:
                # handle unknown labels (e.g., with np.nan)
                filled.append([np.nan, np.nan, np.nan])
        nmat_L_list = np.concatenate((np.array(filled), y.reshape(-1, 1)), axis=1)
        return nmat_L_list
    elif ytarget == "instrument-name":
        row_map = {}
        for row in map_array:
            key = row[1]
            if key not in row_map:
                row_map[key] = row
        
        # Use a list comprehension to build the new array based on the strings in y
        nmat_L_list = [row_map[key] for key in y]
        nmat_L_list =  np.array(nmat_L_list, dtype=object)
    else:
        """
        Generates a NumPy array (nmat_L) from a list of strings (y) by mapping
        the strings to full rows from a map array. The key for the map is formed
        from the first and third columns of map_array. If a combination of the
        first, second, and third columns repeats, only the first occurrence is used.
    
        Args:
            y (list): A list of strings, where each string represents a key for a
                      row in the map_array (e.g., '1_1').
            map_array (np.ndarray): A 2D NumPy array containing the data to map to.
    
        Returns:
            np.ndarray: A new NumPy array (nmat_L) containing the full rows
                        from the map_array.
        """
        # Create a dictionary for fast lookups. We only store the first
        # combination of the first three columns.
        row_map = {}
        for row in map_array:
            key = f"{row[0]}_{row[2]}"
            if key not in row_map:
                row_map[key] = row
        
        # Use a list comprehension to build the new array based on the strings in y
        nmat_L_list = [row_map[key] for key in y]
        nmat_L_list =  np.array(nmat_L_list, dtype=object)
    # Convert the list of arrays into a single NumPy array
    return nmat_L_list

### 16.2.2026
'''
Prompt by GV. in Chat-GPT
given a midi file, create a python function to re-order how the instruments are presented in musescore based on mapping. if an instrument is not present in the midi file but present in mapping, do not show it.
for example, mapping for a given file has the following information:

array([[1, 'Flauti', 0, 73],
[2, 'Oboi', 1, 68],
[3, 'Clarinetti B', 2, 71],
[4, 'Fagotti', 3, 70],
[5, 'Corni C', 4, 60],
[6, 'Trombe C', 5, 56],
[7, 'Timpani', 6, 47],
[8, 'Violini I.', 11, 48],
[9, 'Violini II.', 12, 48],
[10, 'Viole ', 13, 45],
[10, 'Viole ', 13, 48],
[11, 'Celli', 14, 48],
[11, 'Celli', 14, 45],
[12, 'Contrabassi', 15, 48],
[12, 'Contrabassi', 15, 45]], dtype=object)
'''
#from __future__ import annotations

#import numpy as np
#import mido
#from mido import MidiFile, MidiTrack, MetaMessage, Message


def reorder_midi_for_musescore(
    midi_in: str,
    midi_out: str,
    mapping,
    require_notes: bool = True,
):
    """
    Reorder instruments (tracks) as they appear when importing the MIDI into MuseScore,
    based on a mapping table.

    The function:
      - Detects which MIDI channels are present in the file
      - Keeps ONLY channels present in the MIDI (even if mapping contains more)
      - Rebuilds a Type-1 MIDI with:
          Track 0: global meta (tempo/time signature/etc.)
          Track 1..N: one track per channel, in mapping order

    Parameters
    ----------
    midi_in, midi_out : str
        Input/output MIDI file paths.
    mapping : array-like shape (k,4)
        Rows like: [order, display_name, channel, program]
        Example:
            [1, 'Flauti', 0, 73]
            [2, 'Oboi', 1, 68]
            ...
        Channel is 0-15 MIDI channel index.
        Program is GM program number *as you store it*; this function assumes 0-127.
        If your mapping uses 1-128, subtract 1 before calling.
    require_notes : bool
        If True, a channel is considered "present" only if it has note_on/note_off.
        If False, program_change-only channels are also kept.

    Returns
    -------
    kept_channels : list[int]
        Channels that were kept (in final order).
    """
    mapping = np.asarray(mapping, dtype=object)
    if mapping.ndim != 2 or mapping.shape[1] < 4:
        raise ValueError("mapping must be a 2D array with columns [order, name, channel, program].")

    # ---- 1) Build ordered, de-duplicated channel plan from mapping ----
    # mapping columns: 0=order, 1=name, 2=channel, 3=program
    rows = []
    for r in mapping:
        order = int(r[0])
        name = str(r[1])
        ch = int(r[2])
        prog = int(r[3])
        rows.append((order, name, ch, prog))

    # Sort by desired order; if duplicates for same channel exist, keep the first
    rows.sort(key=lambda x: x[0])
    seen = set()
    plan = []
    for order, name, ch, prog in rows:
        if ch in seen:
            continue
        seen.add(ch)
        plan.append((order, name, ch, prog))

    # ---- 2) Parse MIDI and detect channels actually present ----
    mid = MidiFile(midi_in)
    merged = mido.merge_tracks(mid.tracks)

    present_channels = set()
    note_channels = set()

    for msg in merged:
        if msg.is_meta:
            continue
        if hasattr(msg, "channel"):
            present_channels.add(msg.channel)
            if msg.type in ("note_on", "note_off"):
                # note_on with vel 0 is effectively note_off; still counts as note event
                note_channels.add(msg.channel)

    if require_notes:
        present = note_channels
    else:
        present = present_channels

    # Keep only channels that are both in mapping and present in MIDI
    kept_plan = [(order, name, ch, prog) for (order, name, ch, prog) in plan if ch in present]
    kept_channels = [ch for _, _, ch, _ in kept_plan]

    # ---- 3) Collect events with absolute tick times (split by channel) ----
    meta_events = []  # (abs_time, msg)
    chan_events = {ch: [] for ch in kept_channels}  # ch -> list[(abs_time, msg)]

    abs_t = 0
    for msg in merged:
        abs_t += msg.time
        if msg.is_meta:
            # keep global meta in track 0
            # (skip track_name/end_of_track; we'll add our own end_of_track)
            if msg.type not in ("track_name", "end_of_track"):
                meta_events.append((abs_t, msg.copy(time=0)))
        else:
            if hasattr(msg, "channel") and msg.channel in chan_events:
                chan_events[msg.channel].append((abs_t, msg.copy(time=0)))

    # ---- 4) Build output MIDI: track 0 meta + ordered per-channel tracks ----
    out = MidiFile(type=1, ticks_per_beat=mid.ticks_per_beat)

    # Track 0: meta
    meta_track = MidiTrack()
    out.tracks.append(meta_track)

    meta_events.sort(key=lambda x: x[0])
    prev = 0
    for t, msg in meta_events:
        dm = msg.copy(time=t - prev)
        meta_track.append(dm)
        prev = t
    meta_track.append(MetaMessage("end_of_track", time=0))

    # One track per kept channel, in mapping order
    for _, name, ch, prog in kept_plan:
        tr = MidiTrack()
        out.tracks.append(tr)

        # Track name at time 0 (helps MuseScore staff names)
        tr.append(MetaMessage("track_name", name=name, time=0))

        # Ensure program at time 0 (if you want mapping to define instruments)
        tr.append(Message("program_change", channel=ch, program=int(prog), time=0))

        ev = chan_events[ch]
        ev.sort(key=lambda x: x[0])

        prev = 0
        for t, msg in ev:
            tr.append(msg.copy(time=t - prev))
            prev = t

        tr.append(MetaMessage("end_of_track", time=0))

    out.save(midi_out)
    return kept_channels
### 16.2.2026