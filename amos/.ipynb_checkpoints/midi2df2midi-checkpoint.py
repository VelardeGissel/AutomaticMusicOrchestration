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


#df = midi_to_dataframe('midis/sugar-plum-fairy_orch.mid')#violin_flute.mid')#sugar-plum-fairy_orch.mid')
#save_midi_from_df(df, 'midis/output1.mid')

