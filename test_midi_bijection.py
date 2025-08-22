#!/usr/bin/env python
# coding: utf-8

"""
Unit test to verify that midi_to_dataframe and save_midi_from_df form a bijection
and preserve information about multiple instruments of the same type.
"""

import sys
import os
import tempfile
import unittest
import mido

# Add the amos directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'amos'))
from midi2df2midi import midi_to_dataframe, save_midi_from_df


class TestMidiBijection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_midi_path = "midis/sugar-plum-fairy_orch.mid"
        self.temp_dir = tempfile.mkdtemp()
        
    def test_track_channel_preservation(self):
        """Test that track-channel combinations are preserved during conversion."""
        
        # Load original MIDI file
        original_mid = mido.MidiFile(self.test_midi_path)
        
        # Extract original track-channel combinations
        original_track_channels = set()
        original_track_names = {}
        original_programs = {}
        
        for track_idx, track in enumerate(original_mid.tracks):
            track_name = None
            for msg in track:
                if msg.type == 'track_name':
                    track_name = msg.name
                    original_track_names[track_idx] = track_name
                elif msg.type == 'program_change':
                    channel = getattr(msg, 'channel', 0)
                    original_track_channels.add((track_idx, channel))
                    original_programs[(track_idx, channel)] = msg.program
                elif msg.type in ['note_on', 'note_off']:
                    channel = getattr(msg, 'channel', 0)
                    original_track_channels.add((track_idx, channel))
        
        print(f"Original track-channel combinations: {len(original_track_channels)}")
        print("Original tracks with multiple channels:")
        for track_idx, track_name in original_track_names.items():
            channels = [ch for tr, ch in original_track_channels if tr == track_idx]
            if len(channels) > 1:
                print(f"  Track {track_idx} ({track_name}): channels {sorted(channels)}")
        
        # Convert to DataFrame
        df = midi_to_dataframe(self.test_midi_path)
        
        # Extract DataFrame track-channel combinations
        df_track_channels = set(zip(df['track number'], df['channel']))
        
        print(f"DataFrame track-channel combinations: {len(df_track_channels)}")
        
        # Test 1: All original track-channel combinations should be preserved
        missing_combinations = original_track_channels - df_track_channels
        if missing_combinations:
            print(f"Missing track-channel combinations: {missing_combinations}")
        
        self.assertEqual(len(missing_combinations), 0, 
                        f"Lost track-channel combinations: {missing_combinations}")
        
        # Convert back to MIDI
        output_path = os.path.join(self.temp_dir, "reconstructed.mid")
        save_midi_from_df(df, output_path)
        
        # Load reconstructed MIDI
        reconstructed_mid = mido.MidiFile(output_path)
        
        # Extract reconstructed track information
        reconstructed_tracks = {}
        
        for track_idx, track in enumerate(reconstructed_mid.tracks):
            track_name = None
            channels = set()
            programs = {}
            
            for msg in track:
                if msg.type == 'track_name':
                    track_name = msg.name
                elif msg.type == 'program_change':
                    channel = getattr(msg, 'channel', 0)
                    channels.add(channel)
                    programs[channel] = msg.program
                elif msg.type in ['note_on', 'note_off']:
                    channel = getattr(msg, 'channel', 0)
                    channels.add(channel)
            
            if track_name and channels:
                reconstructed_tracks[track_name] = {
                    'channels': channels,
                    'programs': programs
                }
        
        print(f"Reconstructed tracks: {len(reconstructed_tracks)}")
        for track_name, info in reconstructed_tracks.items():
            print(f"  {track_name}: channels {sorted(info['channels'])}")
        
        # Test 2: Check if tracks with multiple instruments are preserved
        multi_instrument_tracks = ["3 Flutes", "2 Oboes", "2 Clarinets in A", "2 Bassoons", "4 Horns in F"]
        
        for track_name in multi_instrument_tracks:
            if track_name in reconstructed_tracks:
                original_channels = []
                for track_idx, name in original_track_names.items():
                    if name == track_name:
                        channels = [ch for tr, ch in original_track_channels if tr == track_idx]
                        original_channels.extend(channels)
                
                reconstructed_channels = list(reconstructed_tracks[track_name]['channels'])
                
                print(f"Track '{track_name}':")
                print(f"  Original channels: {sorted(original_channels)}")
                print(f"  Reconstructed channels: {sorted(reconstructed_channels)}")
                
                # This test will likely fail with current implementation
                self.assertEqual(len(original_channels), len(reconstructed_channels),
                               f"Track '{track_name}' lost channels: {len(original_channels)} -> {len(reconstructed_channels)}")
    
    def test_note_count_preservation(self):
        """Test that the total number of notes is preserved."""
        
        # Count notes in original MIDI
        original_mid = mido.MidiFile(self.test_midi_path)
        original_note_count = 0
        
        for track in original_mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    original_note_count += 1
        
        # Convert to DataFrame and back
        df = midi_to_dataframe(self.test_midi_path)
        df_note_count = len(df)
        
        output_path = os.path.join(self.temp_dir, "reconstructed_notes.mid")
        save_midi_from_df(df, output_path)
        
        # Count notes in reconstructed MIDI
        reconstructed_mid = mido.MidiFile(output_path)
        reconstructed_note_count = 0
        
        for track in reconstructed_mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    reconstructed_note_count += 1
        
        print(f"Original note count: {original_note_count}")
        print(f"DataFrame note count: {df_note_count}")
        print(f"Reconstructed note count: {reconstructed_note_count}")
        
        self.assertEqual(original_note_count, df_note_count, 
                        "Note count differs between original MIDI and DataFrame")
        self.assertEqual(df_note_count, reconstructed_note_count,
                        "Note count differs between DataFrame and reconstructed MIDI")
    
    def test_overlapping_notes_preservation(self):
        """Test that overlapping notes (same pitch, channel, time, different velocities) are preserved."""
        
        # Convert to DataFrame
        df = midi_to_dataframe(self.test_midi_path)
        
        # Find overlapping notes (same onset time, channel, pitch)
        overlapping_groups = df.groupby(['onset in quarter notes', 'channel', 'pitch']).size()
        overlapping_notes = overlapping_groups[overlapping_groups > 1]
        
        print(f"Found {len(overlapping_notes)} sets of overlapping notes")
        
        if len(overlapping_notes) > 0:
            print("Overlapping note details:")
            for (onset, channel, pitch), count in overlapping_notes.items():
                notes = df[(df['onset in quarter notes'] == onset) & 
                          (df['channel'] == channel) & 
                          (df['pitch'] == pitch)]
                velocities = notes['velocity'].tolist()
                print(f"  Onset {onset}, Ch {channel}, Pitch {pitch}: {count} notes with velocities {velocities}")
        
        # Convert back to MIDI and verify overlapping notes are preserved
        output_path = os.path.join(self.temp_dir, "overlapping_test.mid")
        save_midi_from_df(df, output_path)
        
        # Reload and check
        df_reloaded = midi_to_dataframe(output_path)
        overlapping_groups_reloaded = df_reloaded.groupby(['onset in quarter notes', 'channel', 'pitch']).size()
        overlapping_notes_reloaded = overlapping_groups_reloaded[overlapping_groups_reloaded > 1]
        
        print(f"After round-trip: {len(overlapping_notes_reloaded)} sets of overlapping notes")
        
        # The number of overlapping note groups should be preserved
        self.assertEqual(len(overlapping_notes), len(overlapping_notes_reloaded),
                        "Number of overlapping note groups not preserved")
        
        # Total DataFrame size should be preserved
        self.assertEqual(len(df), len(df_reloaded),
                        "Total number of notes not preserved through round-trip")

    def test_bijection_property(self):
        """Test that the conversion forms a proper bijection (reversible without information loss)."""
        
        # Step 1: Original MIDI -> DataFrame
        df1 = midi_to_dataframe(self.test_midi_path)
        
        # Step 2: DataFrame -> MIDI
        temp_midi1 = os.path.join(self.temp_dir, "step1.mid")
        save_midi_from_df(df1, temp_midi1)
        
        # Step 3: MIDI -> DataFrame (second conversion)
        df2 = midi_to_dataframe(temp_midi1)
        
        # Step 4: DataFrame -> MIDI (second round-trip)
        temp_midi2 = os.path.join(self.temp_dir, "step2.mid")
        save_midi_from_df(df2, temp_midi2)
        
        # Step 5: Final DataFrame
        df3 = midi_to_dataframe(temp_midi2)
        
        print(f"Original DataFrame: {len(df1)} notes")
        print(f"After 1st round-trip: {len(df2)} notes")
        print(f"After 2nd round-trip: {len(df3)} notes")
        
        # Test bijection property: multiple round-trips should not change the data
        self.assertEqual(len(df1), len(df2), "Data changed after first round-trip")
        self.assertEqual(len(df2), len(df3), "Data changed after second round-trip")
        
        # Check that key musical information is preserved
        for df_name, df in [("df1", df1), ("df2", df2), ("df3", df3)]:
            unique_pitches = len(df['pitch'].unique())
            unique_channels = len(df['channel'].unique()) 
            unique_tracks = len(df['track number'].unique())
            print(f"{df_name}: {unique_pitches} pitches, {unique_channels} channels, {unique_tracks} tracks")
        
        # These should be identical across all DataFrames
        self.assertEqual(len(df1['pitch'].unique()), len(df2['pitch'].unique()), "Unique pitches changed")
        self.assertEqual(len(df1['channel'].unique()), len(df2['channel'].unique()), "Unique channels changed")
        self.assertEqual(len(df2['pitch'].unique()), len(df3['pitch'].unique()), "Unique pitches changed in 2nd round-trip")
        self.assertEqual(len(df2['channel'].unique()), len(df3['channel'].unique()), "Unique channels changed in 2nd round-trip")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
