#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
14.May.2025 11.54
Prompt by Gissel Velarde
Code generated with https://platform.openai.com/
Date:14.May.2025
Model gpt-4.1
text.format: text
temp: 1.00
tokens: 2048
top_p: 1.00
store: true
prompt:

create a python function using mido to output a pandas dataframe from a midi file. Midi files mostly contain orchestral music and instruments.
The columns of the dataframe should be:
track number (int), track name (string), channel (int), program (int), onset in quarter notes (float), duration in quarter notes (float),  pitch (int), velocity (int).
Instrument name should be read from the midi file if available, if not available it should contain track number.
"""
import mido
import pandas as pd

def midi_to_dataframe(midi_filename):
    mid = mido.MidiFile(midi_filename)

    # Keep track of program changes: (track, channel) -> program
    program_changes = {}
    # Keep track of instrument names: track -> instrument_name
    instrument_names = {}
    # Track names: track -> track_name
    track_names = {}
    # List of note events to be filled
    notes = []
    
    ticks_per_beat = mid.ticks_per_beat

    for i, track in enumerate(mid.tracks):
        cur_time = 0
        # For tracking active notes on each channel - now supports multiple overlapping notes
        active_notes = {}  # (channel, note) -> list of [onset_time, velocity]

        for msg in track:
            cur_time += msg.time
            # Set track name
            if msg.type == 'track_name':
                track_names[i] = msg.name
            elif msg.type == 'instrument_name':
                instrument_names[i] = msg.name
            # Set program change
            elif msg.type == 'program_change':
                program_changes[(i, getattr(msg, 'channel', 0))] = msg.program
            # Note On
            elif msg.type == 'note_on' and msg.velocity > 0:
                k = (msg.channel, msg.note)
                if k not in active_notes:
                    active_notes[k] = []
                # Add this note to the list of active notes
                active_notes[k].append([cur_time, msg.velocity])
            # Note Off or Note On with velocity 0
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active_notes and active_notes[k]:
                    # Remove the oldest note (FIFO - first note on, first note off)
                    onset, velocity = active_notes[k].pop(0)
                    duration = cur_time - onset
                    notes.append({
                        "track number": i,
                        "track name": track_names.get(i, f"Track {i}"),
                        "instrument name": instrument_names.get(i, f"Track {i}"),
                        "channel": msg.channel,
                        "program": program_changes.get((i, msg.channel), 0),
                        "onset in quarter notes": onset / ticks_per_beat,
                        "duration in quarter notes": duration / ticks_per_beat,
                        "pitch": msg.note,
                        "velocity": velocity
                    })
                    # Clean up empty lists
                    if not active_notes[k]:
                        del active_notes[k]
    # Prepare the dataframe
    df = pd.DataFrame(notes)
    # Fallback: if 'instrument name' is not present or empty, use track number
    df['instrument name'] = df['instrument name'].fillna(df['track number'].astype(str))
    # Keep only requested columns, renaming as specified (excluding 'instrument name')
    df = df[["track number", "track name", "channel", "program", "onset in quarter notes",
             "duration in quarter notes", "pitch", "velocity"]]
    return df

# Example usage:
# df = midi_to_dataframe('orchestral_file.mid')
# print(df.head())


# In[2]:


"""
14.May.2025 11.57
Prompt by Gissel Velarde
Code generated  with https://platform.openai.com/
Date:14.May.2025
Model gpt-4.1
text.format: text
temp: 1.00
tokens: 2048
top_p: 1.00
store: true
prompt:
create a python function using mido to save a midi file from a pandas dataframe.
The columns of the dataframe are:
track number (int), track name (string), channel (int), program (int), onset in quarter notes (float), duration in quarter notes (float), pitch (int), velocity (int).
"""
#import pandas as pd
#import mido

def save_midi_from_df(df, output_path, ticks_per_beat=480):
    """
    Save a MIDI file from a pandas DataFrame using mido.
    
    Parameters:
    - df: pandas.DataFrame (columns: 'track number', 'track name', 'channel', 'program',
           'onset in quarter notes', 'duration in quarter notes', 'pitch', 'velocity')
    - output_path: str, where to save the MIDI file.
    - ticks_per_beat: int, MIDI resolution (default 480).
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Group by track number, track name, and channel to preserve multi-instrument tracks
    for (track_num, name, chan), notes in df.groupby(['track number', 'track name', 'channel']):
        # Create a new track for each unique (track_num, name, channel) combination
        track = mido.MidiTrack()
        
        # Convert to native Python types to ensure mido compatibility
        chan = int(chan)
        track_num = int(track_num)
        
        # Set track name - append channel info if there are multiple channels for this track name
        # Check if this track name has multiple channels
        track_channels = df[df['track name'] == name]['channel'].unique()
        if len(track_channels) > 1:
            # Multiple channels for this track name - append channel number
            track_name = f"{name} (Ch {chan})"
        else:
            # Single channel - use original track name
            track_name = name
            
        track.append(mido.MetaMessage('track_name', name=track_name, time=0))
        
        # Set program (instrument)
        prog = int(notes['program'].iloc[0])
        track.append(mido.Message('program_change', program=prog, channel=chan, time=0))

        # Prepare note_on/off events as a list of (time, message)
        events = []
        for _, row in notes.iterrows():
            onset_ticks = int(row['onset in quarter notes'] * ticks_per_beat)
            offset_ticks = int((row['onset in quarter notes'] + row['duration in quarter notes']) * ticks_per_beat)
            pitch = int(row['pitch'])
            velocity = int(row['velocity'])
            chan = int(chan)  # Ensure channel is integer for mido
            # Note on
            events.append((onset_ticks, mido.Message('note_on', note=pitch, velocity=velocity, channel=chan)))
            # Note off
            events.append((offset_ticks, mido.Message('note_off', note=pitch, velocity=0, channel=chan)))
        
        # Sort by absolute time, then add with correct delta times
        events.sort(key=lambda x: x[0])
        last_tick = 0
        for abs_tick, msg in events:
            delta = abs_tick - last_tick
            msg.time = delta
            track.append(msg)
            last_tick = abs_tick
        
        mid.tracks.append(track)

    # Save the MIDI file
    mid.save(output_path)

# Example usage:
# save_midi_from_df(df, 'output.mid')


# In[3]:


#df = midi_to_dataframe('data/samples/midis/sugar-plum-fairy_orch.mid')#violin_flute.mid')#sugar-plum-fairy_orch.mid')
#save_midi_from_df(df, 'runtime/outputs/output1.mid')
'''
Prompt. GV 20.2.2026
Chat-GPT
the python function midi_to_dataframe uses mido to output a pandas dataframe from a MIDI file. 
Currently the dataframe has the following columns:
track number (int), track name (string), channel (int), program (int), onset in quarter notes (float), duration in quarter notes (float),  pitch (int), velocity (int).

update the function midi_to_dataframe so that the following columns are added after column velocity: 
bar number, 
onset inside bar in quarter notes (float),
key signature inside bar,
time signature inside bar,

'''
#import mido
#import pandas as pd

from bisect import bisect_right
import math


def _note_name_to_pc(name: str) -> int:
    """Convert note name like C, Bb, F#, Cb to pitch class 0-11."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Empty note name")

    base_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    letter = name[0].upper()
    if letter not in base_map:
        raise ValueError(f"Bad note letter: {name}")

    pc = base_map[letter]
    acc = name[1:]  # '', '#', 'b', '##', 'bb', etc.
    for ch in acc:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
        else:
            raise ValueError(f"Bad accidental in: {name}")
    return pc % 12


def _parse_key_signature(key: str):
    """
    Parse mido key signature strings like 'C', 'Bb', 'F#m', 'Ebm'.
    Return (tonic_pc, mode, prefer_flats_bool).
    """
    key = (key or "C").strip()
    mode = "minor" if key.endswith("m") else "major"
    tonic = key[:-1] if mode == "minor" else key

    prefer_flats = ("b" in tonic)
    tonic_pc = _note_name_to_pc(tonic)
    return tonic_pc, mode, prefer_flats


def _pc_to_key_signature(pc: int, mode: str, prefer_flats: bool) -> str:
    """Convert tonic pitch class + mode back to a key signature string."""
    pc %= 12
    major_sharp = {
        0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
        6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
    }
    major_flat = {
        0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
        6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"
    }
    tonic = (major_flat if prefer_flats else major_sharp)[pc]
    return tonic + ("m" if mode == "minor" else "")


def _transpose_key_signature(key: str, semitones_up: int) -> str:
    """Transpose a key signature by semitones_up (written key = concert key + transposition)."""
    semitones_up %= 12
    if semitones_up == 0:
        return key or "C"
    tonic_pc, mode, prefer_flats = _parse_key_signature(key or "C")
    new_pc = (tonic_pc + semitones_up) % 12
    return _pc_to_key_signature(new_pc, mode, prefer_flats)


def _infer_transposition_semitones(track_name: str, instrument_name: str) -> int:
    """
    Infer transposition (in semitones) from track/instrument name text.
    Convention: written key = concert key + semitones.
    """
    text = f"{track_name or ''} {instrument_name or ''}".lower()

    # English Horn / Cor Anglais: sounds P5 lower => written is +7
    if "english horn" in text or "cor anglais" in text or "cor anglai" in text:
        return 7

    # Horn in F: sounds P5 lower => written is +7
    if "horn in f" in text or "f horn" in text or ("horn" in text and "in f" in text):
        return 7
    if text.strip() in {"horn", "horns"}:
        return 7

    # Clarinet in A: sounds m3 lower => written is +3
    if "clarinet" in text and " in a" in text:
        return 3

    # Clarinet in Bb / B-flat: sounds M2 lower => written is +2
    if "clarinet" in text and ("bb" in text or "b♭" in text or "b-flat" in text or " in b" in text):
        return 2

    # Optional: Trumpet in Bb: +2
    if "trumpet" in text and ("bb" in text or "b♭" in text or "b-flat" in text or " in b" in text):
        return 2

    return 0


def midi_to_dataframe_ext(midi_filename):
    mid = mido.MidiFile(midi_filename)
    ticks_per_beat = mid.ticks_per_beat

    # -----------------------------
    # Global timelines for time signature and concert key signature
    # -----------------------------
    merged = mido.merge_tracks(mid.tracks)

    time_sig_changes = [(0, 4, 4)]  # (abs_tick, numerator, denominator)
    key_sig_changes = [(0, "C")]    # (abs_tick, key_str)

    abs_tick = 0
    for msg in merged:
        abs_tick += msg.time
        if msg.type == "time_signature":
            time_sig_changes.append((abs_tick, msg.numerator, msg.denominator))
        elif msg.type == "key_signature":
            key_sig_changes.append((abs_tick, msg.key))

    time_sig_changes.sort(key=lambda x: x[0])
    key_sig_changes.sort(key=lambda x: x[0])

    # Build time-signature regions for bar calculations
    regions = []
    current_bar = 1
    for idx, (t0, n, d) in enumerate(time_sig_changes):
        bar_len_ticks = ticks_per_beat * n * (4.0 / d)
        regions.append({
            "tick": t0,
            "num": n,
            "den": d,
            "bar_len_ticks": bar_len_ticks,
            "start_bar": current_bar
        })

        if idx + 1 < len(time_sig_changes):
            t_next = time_sig_changes[idx + 1][0]
            L = max(0, t_next - t0)
            if L == 0:
                bars_started = 0
            else:
                bars_started = int(math.floor((L - 1) / bar_len_ticks)) + 1
            current_bar += bars_started

    region_ticks = [r["tick"] for r in regions]
    key_ticks = [t for (t, _) in key_sig_changes]

    def lookup_time_sig_region(tick):
        i = bisect_right(region_ticks, tick) - 1
        if i < 0:
            i = 0
        return regions[i]

    def lookup_concert_key(tick):
        j = bisect_right(key_ticks, tick) - 1
        if j < 0:
            j = 0
        return key_sig_changes[j][1] or "C"

    # -----------------------------
    # Original parsing logic + new columns
    # -----------------------------
    program_changes = {}   # (track, channel) -> program
    instrument_names = {}  # track -> instrument_name
    track_names = {}       # track -> track_name
    notes = []

    for i, track in enumerate(mid.tracks):
        cur_time = 0
        active_notes = {}  # (channel, note) -> list of [onset_time, velocity]

        for msg in track:
            cur_time += msg.time

            if msg.type == 'track_name':
                track_names[i] = msg.name
            elif msg.type == 'instrument_name':
                instrument_names[i] = msg.name
            elif msg.type == 'program_change':
                program_changes[(i, getattr(msg, 'channel', 0))] = msg.program

            elif msg.type == 'note_on' and msg.velocity > 0:
                k = (msg.channel, msg.note)
                active_notes.setdefault(k, []).append([cur_time, msg.velocity])

            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active_notes and active_notes[k]:
                    onset, velocity = active_notes[k].pop(0)
                    duration = cur_time - onset

                    # Bar/time signature at note onset
                    region = lookup_time_sig_region(onset)
                    beats_in_bar = int(region["num"])
                    unit_in_bar = int(region["den"])

                    offset = onset - region["tick"]
                    bar_len = region["bar_len_ticks"]
                    bar_number = int(region["start_bar"] + math.floor(offset / bar_len))
                    onset_in_bar_qn = float((offset % bar_len) / ticks_per_beat)

                    # Pitch class inside bar = tonic pitch class of (written) key at onset
                    concert_key = lookup_concert_key(onset)
                    transposition = _infer_transposition_semitones(
                        track_names.get(i, f"Track {i}"),
                        instrument_names.get(i, f"Track {i}")
                    )
                    written_key = _transpose_key_signature(concert_key, transposition)
                    tonic_pc, _, _ = _parse_key_signature(written_key)
                    pitch_class_inside_bar = int(tonic_pc)

                    # Pitch-derived pitch class and octave
                    pitch = int(msg.note)
                    pitch_class_pitch = int(pitch % 12)
                    pitch_class_pitch_octave = int((pitch // 12) - 1)  # C4=60 -> 4

                    notes.append({
                        "track number": i,
                        "track name": track_names.get(i, f"Track {i}"),
                        "instrument name": instrument_names.get(i, f"Track {i}"),
                        "channel": msg.channel,
                        "program": program_changes.get((i, msg.channel), 0),
                        "onset in quarter notes": onset / ticks_per_beat,
                        "duration in quarter notes": duration / ticks_per_beat,
                        "pitch": pitch,
                        "velocity": velocity,

                        "bar number": bar_number,
                        "onset inside bar in quarter notes": onset_in_bar_qn,
                        "pitch class inside bar": pitch_class_inside_bar,
                        "pitch class pitch": pitch_class_pitch,
                        "pitch class pitch octave": pitch_class_pitch_octave,

                        "beats in bar": beats_in_bar,
                        "unit in bar": unit_in_bar,
                    })

                    if not active_notes[k]:
                        del active_notes[k]

    # -----------------------------
    # DataFrame + enforce exact column order
    # -----------------------------
    df = pd.DataFrame(notes)

    if not df.empty:
        df['instrument name'] = df['instrument name'].fillna(df['track number'].astype(str))

    df = df[
        [
            "track number",
            "track name",
            "channel",
            "program",
            "onset in quarter notes",
            "duration in quarter notes",
            "pitch",
            "velocity",
            "bar number",
            "onset inside bar in quarter notes",
            "pitch class inside bar",
            "pitch class pitch",
            "pitch class pitch octave",
            "beats in bar",
            "unit in bar",
        ]
    ]

    return df
###
## GV. 20.2.2026
#updated save function

###
#import mido
#import pandas as pd
from collections import Counter


def save_midi_from_df_ext(df, output_path, ticks_per_beat=480):
    """
    Save a MIDI file from a pandas DataFrame (as produced by midi_to_dataframe) using mido.

    Required columns (minimum):
      - 'track number', 'track name', 'channel', 'program',
        'onset in quarter notes', 'duration in quarter notes', 'pitch', 'velocity'

    Optional columns (if present, used to write global meta messages):
      - 'beats in bar', 'unit in bar'                  -> time_signature meta messages
      - 'pitch class inside bar' (0-11)                -> key_signature meta messages

    Notes
    -----
    - MuseScore recognizes key signature only if MIDI key_signature meta events exist.
    - This function writes a conductor/meta track (track 0) containing those meta events.
    """
    ticks_per_beat = int(ticks_per_beat)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    if df is None or len(df) == 0:
        mid.save(output_path)
        return

    required = {
        "track number", "track name", "channel", "program",
        "onset in quarter notes", "duration in quarter notes",
        "pitch", "velocity",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    # --- helpers ---
    # Use common flat spellings for the "black keys" for readability in notation software.
    pc_to_major_key = {
        0: "C",
        1: "Db",
        2: "D",
        3: "Eb",
        4: "E",
        5: "F",
        6: "Gb",
        7: "G",
        8: "Ab",
        9: "A",
        10: "Bb",
        11: "B",
    }

    def qn_to_ticks(qn):
        return int(round(float(qn) * ticks_per_beat))

    # -----------------------------
    # 1) Conductor / meta track
    # -----------------------------
    meta_track = mido.MidiTrack()

    # Collect meta events as (abs_tick, kind, payload_tuple)
    # then sort and convert to delta times.
    meta_events = []

    # (a) Time signatures
    if "beats in bar" in df.columns and "unit in bar" in df.columns:
        ts_df = df[["onset in quarter notes", "beats in bar", "unit in bar"]].dropna().copy()
        if not ts_df.empty:
            ts_df["abs_tick"] = ts_df["onset in quarter notes"].map(qn_to_ticks)
            ts_df["beats in bar"] = ts_df["beats in bar"].astype(int)
            ts_df["unit in bar"] = ts_df["unit in bar"].astype(int)
            ts_df = ts_df.sort_values(["abs_tick"])

            # one time signature per tick; keep first deterministic
            ts_df = ts_df.drop_duplicates(subset=["abs_tick"], keep="first")

            # compress consecutive duplicates
            last = None
            for abs_tick, n, d in ts_df[["abs_tick", "beats in bar", "unit in bar"]].itertuples(index=False, name=None):
                cur = (int(n), int(d))
                if last != cur:
                    meta_events.append((int(abs_tick), "time_signature", cur))
                    last = cur

    # Ensure time signature at tick 0 (default 4/4 if nothing present)
    has_ts_at_0 = any(t == 0 and k == "time_signature" for (t, k, _) in meta_events)
    if not has_ts_at_0:
        # If we already have some TS later, use the first known; else default 4/4.
        future_ts = [ev for ev in meta_events if ev[1] == "time_signature"]
        if future_ts:
            _, _, (n, d) = min(future_ts, key=lambda x: x[0])
            meta_events.append((0, "time_signature", (n, d)))
        else:
            meta_events.append((0, "time_signature", (4, 4)))

    # (b) Key signatures from pitch class inside bar
    if "pitch class inside bar" in df.columns:
        ks_df = df[["onset in quarter notes", "pitch class inside bar"]].dropna().copy()
        if not ks_df.empty:
            ks_df["abs_tick"] = ks_df["onset in quarter notes"].map(qn_to_ticks)
            ks_df["pc"] = ks_df["pitch class inside bar"].astype(int) % 12
            ks_df = ks_df.sort_values(["abs_tick"])

            # For each tick, choose the most common pitch class across notes at that tick
            last_key = None
            for abs_tick, group in ks_df.groupby("abs_tick", sort=True):
                pcs = group["pc"].tolist()
                if not pcs:
                    continue
                pc_mode = Counter(pcs).most_common(1)[0][0]
                key_str = pc_to_major_key.get(int(pc_mode), "C")
                if key_str != last_key:
                    meta_events.append((int(abs_tick), "key_signature", (key_str,)))
                    last_key = key_str

    # Ensure key signature at tick 0 (default C if nothing present)
    has_ks_at_0 = any(t == 0 and k == "key_signature" for (t, k, _) in meta_events)
    if not has_ks_at_0:
        future_ks = [ev for ev in meta_events if ev[1] == "key_signature"]
        if future_ks:
            _, _, (key_str,) = min(future_ks, key=lambda x: x[0])
            meta_events.append((0, "key_signature", (key_str,)))
        else:
            meta_events.append((0, "key_signature", ("C",)))

    # Sort meta events; for same tick, place time_signature before key_signature (either is fine)
    kind_order = {"time_signature": 0, "key_signature": 1}
    meta_events.sort(key=lambda x: (x[0], kind_order.get(x[1], 99)))

    # Convert to delta times and write into meta track
    last_tick = 0
    for abs_tick, kind, payload in meta_events:
        delta = abs_tick - last_tick
        if kind == "time_signature":
            n, d = payload
            meta_track.append(
                mido.MetaMessage(
                    "time_signature",
                    numerator=int(n),
                    denominator=int(d),
                    clocks_per_click=24,
                    notated_32nd_notes_per_beat=8,
                    time=int(delta),
                )
            )
        elif kind == "key_signature":
            (key_str,) = payload
            meta_track.append(mido.MetaMessage("key_signature", key=str(key_str), time=int(delta)))
        last_tick = abs_tick

    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta_track)

    # -----------------------------
    # 2) Note tracks
    # -----------------------------
    group_cols = ["track number", "track name", "channel"]
    for (track_num, name, chan), notes in df.groupby(group_cols, sort=True):
        track = mido.MidiTrack()

        chan = int(chan)
        track_num = int(track_num)
        track_name = str(name) if pd.notna(name) else f"Track {track_num}"

        # Append channel info if this track name uses multiple channels
        track_channels = df[df["track name"] == name]["channel"].dropna().unique()
        if len(track_channels) > 1:
            track_name_to_write = f"{track_name} (Ch {chan})"
        else:
            track_name_to_write = track_name

        track.append(mido.MetaMessage("track_name", name=track_name_to_write, time=0))

        prog = int(notes["program"].iloc[0]) if len(notes) else 0
        track.append(mido.Message("program_change", program=prog, channel=chan, time=0))

        events = []
        for _, row in notes.iterrows():
            onset_ticks = qn_to_ticks(row["onset in quarter notes"])
            offset_ticks = qn_to_ticks(row["onset in quarter notes"] + row["duration in quarter notes"])
            pitch = int(row["pitch"])
            velocity = int(row["velocity"])

            events.append((onset_ticks, mido.Message("note_on", note=pitch, velocity=velocity, channel=chan)))
            events.append((offset_ticks, mido.Message("note_off", note=pitch, velocity=0, channel=chan)))

        # Sort by time; for same time, note_off before note_on to reduce stuck notes
        def _event_sort_key(item):
            abs_t, msg = item
            order = 0 if msg.type == "note_off" else 1
            return (abs_t, order, getattr(msg, "note", 0))

        events.sort(key=_event_sort_key)

        last_tick = 0
        for abs_tick, msg in events:
            msg.time = int(abs_tick - last_tick)
            track.append(msg)
            last_tick = abs_tick

        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(track)

    mid.save(output_path)