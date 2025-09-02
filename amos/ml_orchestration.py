#!/usr/bin/env python
# coding: utf-8

"""
Machine Learning Orchestration Module
By Gissel Velarde
Date: September 2025

This module contains functions for automatic music orchestration using machine learning.
"""

import numpy as np
import pandas as pd
import time
import mido
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Import from local modules
from midi2df2midi import midi_to_dataframe, save_midi_from_df
from mappings import fill_quaterna_columns, learn_quaterna_mapping

# Keras/TensorFlow imports for neural networks
try:
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, LayerNormalization, MultiHeadAttention
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. LSTM and Transformer classifiers will be disabled.")


# === Keras Model Wrappers and Builders ===

if KERAS_AVAILABLE:
    class KerasClassifierWrapper:
        """
        A wrapper class to make a Keras model compatible with scikit-learn's
        pipeline and plotting functions.
        """
        def __init__(self, model_builder_func, **kwargs):
            self.model = None
            self.model_builder_func = model_builder_func
            self.builder_kwargs = kwargs
            self.input_shape = None
            self.classes_ = None

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            num_classes = len(self.classes_)
            self.input_shape = (1, X.shape[1])
            X_reshaped = np.expand_dims(X, axis=1)
            self.model = self.model_builder_func(self.input_shape, num_classes, **self.builder_kwargs)
            y_one_hot = to_categorical(y, num_classes=num_classes)
            self.model.fit(X_reshaped, y_one_hot, epochs=50, batch_size=32, verbose=0)
            return self

        def predict_proba(self, X):
            X_reshaped = np.expand_dims(X, axis=1)
            return self.model.predict(X_reshaped, verbose=0)

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)
        
        def score(self, X, y):
            X_reshaped = np.expand_dims(X, axis=1)
            y_one_hot = to_categorical(y, num_classes=len(self.classes_))
            _, accuracy = self.model.evaluate(X_reshaped, y_one_hot, verbose=0)
            return accuracy

    class TransformerBlock(tf.keras.layers.Layer):
        """A custom Keras layer for a simple Transformer block."""
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
            super(TransformerBlock, self).__init__(**kwargs)
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential([
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim)
            ])
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.rate = rate

        def call(self, inputs, training=False):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

        def get_config(self):
            config = super(TransformerBlock, self).get_config()
            config.update({
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate
            })
            return config

    def build_lstm_classifier(input_shape, num_classes):
        """Builds a three-layer stacked LSTM model."""
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(128)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    def build_transformer_classifier(input_shape, num_classes):
        """Builds a simple Transformer model."""
        embed_dim = 16
        num_heads = 2
        ff_dim = 16
        inputs = Input(shape=input_shape)
        x = Dense(embed_dim)(inputs)
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(8, activation="relu")(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model


# === Helper Functions ===

def defineXy(nmat, ytarget="track-channel"):
    """Extract features and labels from note matrix."""
    if ytarget == "program":
        X = nmat[:, 4:8]  # onset, duration, pitch, velocity
        y = nmat[:, 3]    # program
    else:
        # X: onset, duration, pitch, velocity, y=track_channel
        X = nmat[:, 4:8]  # onset, duration, pitch, velocity
        A = nmat[:, 0].astype(str)
        B = nmat[:, 2].astype(str)
        y = np.char.add(np.char.add(A, '_'), B)  # Label is "track_channel"
    return X, y


def clf_predict(X2, le, model, mapping, ytarget):
    """Predict orchestration using trained model."""
    y_pred = model.predict(X2)
    print("Predictions ", np.unique(y_pred))
    # inverse transform to obtained the original labels:
    y_pred_orig = le.inverse_transform(y_pred) #track_channel
    # Fill columns track, track name, channel, program using the mapping
    new_cols = fill_quaterna_columns(y_pred_orig, mapping, ytarget)
    print("Predictions map", np.unique(y_pred_orig))
    nmat = np.concatenate((new_cols, X2), axis=1)
    return nmat


def split_and_encode(X, y, test_size=0.2, random_state=42):
    """Split data and encode labels."""
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    elif test_size == 0:
        X_train = X
        y_train = y
        X_test, y_test = 0, 0  # We will not use X_test, y_test for inference
    
    le = LabelEncoder()
    le.fit(y_train)  # Fit only on train 
    print("y_train, test size:", test_size, ", labels:", np.unique(y_train))
    y_train = le.transform(y_train)  # Will be 0,1,...,N-1
    if test_size > 0:
        y_test = le.transform(y_test)
    return X_train, X_test, y_train, y_test, le


def save_midi_from_df_with_timing(df, output_path, reference_midi_path=None, ticks_per_beat=480):
    """
    Enhanced MIDI saving that preserves musical structure from reference file.
    CORRECTED VERSION: Adds only ONE time signature to avoid conflicts.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    
    # Extract the MAIN time signature and tempo (not all variations)
    main_time_signature = None
    main_tempo = None
    
    if reference_midi_path:
        try:
            ref_mid = mido.MidiFile(reference_midi_path)
            print(f"Analyzing musical structure of {reference_midi_path}")
            
            # Find time signatures and tempos
            time_signatures = []
            tempos = []
            
            for track in ref_mid.tracks:
                for msg in track:
                    if msg.type == 'time_signature':
                        time_signatures.append(msg)
                    elif msg.type == 'set_tempo':
                        tempos.append(msg)
            
            # For Für Elise, prioritize 3/8 time signature
            print(f"Found {len(time_signatures)} time signatures")
            for ts in time_signatures:
                if ts.numerator == 3 and ts.denominator == 8:
                    main_time_signature = ts
                    print(f"✓ Selected main time signature: 3/8")
                    break
            
            # If no 3/8 found, use the first time signature
            if main_time_signature is None and time_signatures:
                main_time_signature = time_signatures[0]
                print(f"✓ No 3/8 found, using first: {main_time_signature.numerator}/{main_time_signature.denominator}")
            
            # Use the first tempo found
            if tempos:
                main_tempo = tempos[0]
                bpm = mido.tempo2bpm(main_tempo.tempo)
                print(f"✓ Using tempo: {bpm:.1f} BPM")
                
        except Exception as e:
            print(f"Could not extract structure from reference: {e}")
    
    # Group by track number, track name, and channel
    track_added_meta = False  # Flag to ensure we only add meta info once
    
    for track_idx, ((track_num, name, chan), notes) in enumerate(df.groupby(['track number', 'track name', 'channel'])):
        track = mido.MidiTrack()
        
        chan = int(chan)
        track_num = int(track_num)
        
        # Set track name
        track_channels = df[df['track name'] == name]['channel'].unique()
        if len(track_channels) > 1:
            track_name = f"{name} (Ch {chan})"
        else:
            track_name = name
            
        track.append(mido.MetaMessage('track_name', name=track_name, time=0))
        
        # Add tempo and time signature ONLY ONCE to the first track
        if not track_added_meta:
            if main_tempo:
                track.append(mido.MetaMessage('set_tempo', tempo=main_tempo.tempo, time=0))
            
            if main_time_signature:
                track.append(mido.MetaMessage('time_signature',
                                            numerator=main_time_signature.numerator,
                                            denominator=main_time_signature.denominator,
                                            clocks_per_click=main_time_signature.clocks_per_click,
                                            notated_32nd_notes_per_beat=main_time_signature.notated_32nd_notes_per_beat,
                                            time=0))
                print(f"✓ Added single time signature: {main_time_signature.numerator}/{main_time_signature.denominator}")
            
            track_added_meta = True
        
        # Set program
        prog = int(notes['program'].iloc[0])
        track.append(mido.Message('program_change', program=prog, channel=chan, time=0))
        
        # Prepare note events
        events = []
        for _, row in notes.iterrows():
            onset_ticks = int(row['onset in quarter notes'] * ticks_per_beat)
            offset_ticks = int((row['onset in quarter notes'] + row['duration in quarter notes']) * ticks_per_beat)
            pitch = int(row['pitch'])
            velocity = int(row['velocity'])
            
            events.append((onset_ticks, mido.Message('note_on', note=pitch, velocity=velocity, channel=chan)))
            events.append((offset_ticks, mido.Message('note_off', note=pitch, velocity=0, channel=chan)))
        
        # Sort and add events with correct delta times
        events.sort(key=lambda x: x[0])
        last_tick = 0
        for abs_tick, msg in events:
            delta = abs_tick - last_tick
            msg.time = delta
            track.append(msg)
            last_tick = abs_tick
        
        mid.tracks.append(track)
    
    mid.save(output_path)
    print(f"Saved {output_path} with preserved musical structure")


# === Main ML Orchestration Functions ===

def ml_exp(filein, fileout, ytarget="track-channel"):
    """
    Original ML orchestration function (without time signature preservation).
    """
    names = [
        "XGBoost",
        "Random_Forest",
        "Decision_Tree",
        "LSTM Classifier" if KERAS_AVAILABLE else None,
        "Nearest_Neighbors ",
        "MLP3",   
        "Transformer Classifier" if KERAS_AVAILABLE else None,
        "Naive_Bayes",
        "MLP1",
        "AdaBoost",         
    ]
    
    classifiers = [
        XGBClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KerasClassifierWrapper(build_lstm_classifier) if KERAS_AVAILABLE else None,
        KNeighborsClassifier(1),
        MLPClassifier(hidden_layer_sizes=(128,128,128)),
        KerasClassifierWrapper(build_transformer_classifier) if KERAS_AVAILABLE else None,
        GaussianNB(),
        MLPClassifier(),
        AdaBoostClassifier(),         
    ]
    
    # Filter out None values if Keras is not available
    names = [name for name in names if name is not None]
    classifiers = [clf for clf in classifiers if clf is not None]
    
    # Load and process source file
    dfnmat = midi_to_dataframe(filein)
    dfnmat = dfnmat.sort_values(
        ['onset in quarter notes','duration in quarter notes', 'track number'],
        ascending=[True, True, True]
    )
    nmat = dfnmat.to_numpy()  
    mapping = learn_quaterna_mapping(nmat, ytarget)
    print("mapping", mapping)
    
    X, y = defineXy(nmat, ytarget)
    print("Labels", np.unique(y))
    print("Number of events in ", filein, ":", X.shape[0])
    print("last onset at ", X[X.shape[0]-1, 0])
    
    # Load and process target file
    dfnmat = midi_to_dataframe(fileout) 
    dfnmat = dfnmat.sort_values(
        ['onset in quarter notes','duration in quarter notes', 'track number'],
        ascending=[True, True, True]
    )
    nmat2 = dfnmat.to_numpy()
    X2 = nmat2[:,4:8] #onset, duration, pitch, velocity
    print("Number of events in", fileout, ":", X2.shape[0])
    print("last onset at ", X2[X2.shape[0]-1,0])
    
    # Partition the dataset
    X_train, X_test, y_train, y_test, le = split_and_encode(X, y, test_size=0.2, random_state=42)
    X_train_f, X_test_f, y_train_f, y_test_f, le_f = split_and_encode(X, y, test_size=0, random_state=42)
    
    # Train and predict with each classifier
    for name, clf in zip(names, classifiers):
        start = time.time()
        
        if name in ["LSTM Classifier", "Transformer Classifier"]:
            clf_pipeline = make_pipeline(StandardScaler(), clf)
        else:
            clf_pipeline = clf
            
        clf_pipeline.fit(X_train, y_train)
        score = clf_pipeline.score(X_test, y_test)
        end = time.time()
        
        print("---------", name)
        print("Train Time (sec) :", f"{end - start:.4f}")
        print("Score on Test (20%): ", f"{score:.4f}") 
        
        # Predict orchestration 
        clf_pipeline.fit(X_train_f, y_train_f)
        data = clf_predict(X2, le_f, clf_pipeline, mapping, ytarget)
        
        # Convert to DataFrame
        dfdata = pd.DataFrame(data, columns=[
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
        
        extensions = name + ".mid"
        filename = fileout.replace(".mid", extensions)
        print('Orchestration:', filename)
        save_midi_from_df(dfdata, filename)


def ml_exp_with_timing(filein, fileout, ytarget="track-channel"):
    """
    Enhanced ML orchestration function that preserves time signatures and tempo.
    """
    names = [
        "XGBoost",
        "Random_Forest",
        "Decision_Tree",
        "LSTM Classifier" if KERAS_AVAILABLE else None,
        "Nearest_Neighbors ",
        "MLP3",   
        "Transformer Classifier" if KERAS_AVAILABLE else None,
        "Naive_Bayes",
        "MLP1",
        "AdaBoost",         
    ]
    
    classifiers = [
        XGBClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KerasClassifierWrapper(build_lstm_classifier) if KERAS_AVAILABLE else None,
        KNeighborsClassifier(1),
        MLPClassifier(hidden_layer_sizes=(128,128,128)),
        KerasClassifierWrapper(build_transformer_classifier) if KERAS_AVAILABLE else None,
        GaussianNB(),
        MLPClassifier(),
        AdaBoostClassifier(),         
    ]
    
    # Filter out None values if Keras is not available
    names = [name for name in names if name is not None]
    classifiers = [clf for clf in classifiers if clf is not None]
    
    # Load and process source file
    print(f"Learning orchestration style from: {filein}")
    dfnmat = midi_to_dataframe(filein)
    dfnmat = dfnmat.sort_values(
        ['onset in quarter notes','duration in quarter notes', 'track number'],
        ascending=[True, True, True]
    )
    nmat = dfnmat.to_numpy()  
    mapping = learn_quaterna_mapping(nmat, ytarget)
    print("Mapping:", mapping)
    
    X, y = defineXy(nmat, ytarget)
    print("Labels", np.unique(y))
    print("Number of events in", filein, ":", X.shape[0])
    print("Last onset at", X[X.shape[0]-1, 0])
    
    # Load and process target file
    print(f"\\nProcessing target file: {fileout}")
    dfnmat2 = midi_to_dataframe(fileout) 
    dfnmat2 = dfnmat2.sort_values(
        ['onset in quarter notes','duration in quarter notes', 'track number'],
        ascending=[True, True, True]
    )
    nmat2 = dfnmat2.to_numpy()
    X2 = nmat2[:,4:8] #onset, duration, pitch, velocity
    print("Number of events in", fileout, ":", X2.shape[0])
    print("Last onset at", X2[X2.shape[0]-1,0])
    
    # Get original ticks_per_beat for precise timing
    try:
        original_midi = mido.MidiFile(fileout)
        original_ticks_per_beat = original_midi.ticks_per_beat
        print(f"Original ticks_per_beat: {original_ticks_per_beat}")
    except:
        original_ticks_per_beat = 480
    
    # Partition the dataset
    X_train, X_test, y_train, y_test, le = split_and_encode(X, y, test_size=0.2, random_state=42)
    X_train_f, X_test_f, y_train_f, y_test_f, le_f = split_and_encode(X, y, test_size=0, random_state=42)
    
    # Train and predict with each classifier
    for name, clf in zip(names, classifiers):
        print(f"\\n--------- {name} ---------")
        start = time.time()
        
        if name in ["LSTM Classifier", "Transformer Classifier"]:
            clf_pipeline = make_pipeline(StandardScaler(), clf)
        else:
            clf_pipeline = clf
            
        clf_pipeline.fit(X_train, y_train)
        score = clf_pipeline.score(X_test, y_test)
        end = time.time()
        
        print("Train Time (sec):", f"{end - start:.4f}")
        print("Score on Test (20%):", f"{score:.4f}")
        
        # Predict orchestration 
        clf_pipeline.fit(X_train_f, y_train_f)
        data = clf_predict(X2, le_f, clf_pipeline, mapping, ytarget)
        
        # Convert to DataFrame
        dfdata = pd.DataFrame(data, columns=[
            'track number', 'track name', 'channel', 'program',
            'onset in quarter notes', 'duration in quarter notes', 'pitch', 'velocity'
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
        
        # Save with preserved musical structure
        extensions = name + "_WITH_TIMING.mid"
        filename = fileout.replace(".mid", extensions)
        print('Orchestration with preserved structure:', filename)
        
        # Use EXACT timing preservation function that preserves ALL timing events
        save_midi_with_exact_timing_structure(dfdata, filename, 
                                             reference_midi_path=fileout)


# === Advanced Timing Preservation Functions ===

def extract_timing_structure(reference_midi_path):
    """
    Extract the complete timing structure from a reference MIDI file.
    Returns a list of timing events with their absolute positions.
    """
    try:
        ref_mid = mido.MidiFile(reference_midi_path)
        print(f"Extracting timing structure from {reference_midi_path}")
        print(f"ticks_per_beat: {ref_mid.ticks_per_beat}")
        
        timing_events = []
        
        for track_idx, track in enumerate(ref_mid.tracks):
            current_time = 0
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'time_signature':
                    timing_events.append({
                        'type': 'time_signature',
                        'time_ticks': current_time,
                        'time_quarters': current_time / ref_mid.ticks_per_beat,
                        'track': track_idx,
                        'message': msg,
                        'display': f"{msg.numerator}/{msg.denominator}"
                    })
                    
                elif msg.type == 'set_tempo':
                    timing_events.append({
                        'type': 'tempo',
                        'time_ticks': current_time,
                        'time_quarters': current_time / ref_mid.ticks_per_beat,
                        'track': track_idx,
                        'message': msg,
                        'display': f"{mido.tempo2bpm(msg.tempo):.1f} BPM"
                    })
        
        # Sort by time
        timing_events.sort(key=lambda x: x['time_ticks'])
        
        print(f"Found {len(timing_events)} timing events:")
        for i, event in enumerate(timing_events):
            print(f"  {i+1}. {event['type']}: {event['display']} at {event['time_quarters']:.3f} quarters ({event['time_ticks']} ticks)")
        
        return timing_events, ref_mid.ticks_per_beat
        
    except Exception as e:
        print(f"Error extracting timing structure: {e}")
        return [], 480


def save_midi_with_exact_timing_structure(df, output_path, reference_midi_path, target_ticks_per_beat=None):
    """
    Save MIDI with EXACT timing structure preserved from reference file.
    This preserves all time signature and tempo changes at their correct positions.
    """
    print(f"\n=== SAVING WITH EXACT TIMING STRUCTURE ===")
    
    # Extract timing structure from reference
    timing_events, ref_ticks_per_beat = extract_timing_structure(reference_midi_path)
    
    # Use target ticks_per_beat if specified, otherwise use reference
    if target_ticks_per_beat is None:
        target_ticks_per_beat = ref_ticks_per_beat
    
    print(f"Using ticks_per_beat: {target_ticks_per_beat}")
    
    # Create new MIDI file
    mid = mido.MidiFile(ticks_per_beat=target_ticks_per_beat)
    
    # Scale timing events if ticks_per_beat is different
    scale_factor = target_ticks_per_beat / ref_ticks_per_beat if ref_ticks_per_beat != target_ticks_per_beat else 1.0
    
    if scale_factor != 1.0:
        print(f"Scaling timing events by factor: {scale_factor}")
        for event in timing_events:
            event['time_ticks'] = int(event['time_ticks'] * scale_factor)
    
    # Group notes by track
    note_tracks = []
    for (track_num, name, chan), notes in df.groupby(['track number', 'track name', 'channel']):
        note_tracks.append({
            'track_num': int(track_num),
            'name': name,
            'channel': int(chan),
            'notes': notes
        })
    
    # Create conductor track with timing structure
    conductor_track = mido.MidiTrack()
    conductor_track.append(mido.MetaMessage('track_name', name='Conductor', time=0))
    
    # Add all timing events to conductor track
    last_time = 0
    for event in timing_events:
        delta_time = event['time_ticks'] - last_time
        
        if event['type'] == 'time_signature':
            msg = event['message']
            conductor_track.append(mido.MetaMessage('time_signature',
                                                  numerator=msg.numerator,
                                                  denominator=msg.denominator,
                                                  clocks_per_click=msg.clocks_per_click,
                                                  notated_32nd_notes_per_beat=msg.notated_32nd_notes_per_beat,
                                                  time=delta_time))
            print(f"✓ Added {msg.numerator}/{msg.denominator} at {event['time_quarters']:.3f} quarters")
            
        elif event['type'] == 'tempo':
            msg = event['message']
            conductor_track.append(mido.MetaMessage('set_tempo',
                                                  tempo=msg.tempo,
                                                  time=delta_time))
            bpm = mido.tempo2bpm(msg.tempo)
            print(f"✓ Added tempo {bpm:.1f} BPM at {event['time_quarters']:.3f} quarters")
        
        last_time = event['time_ticks']
    
    mid.tracks.append(conductor_track)
    
    # Create note tracks
    for track_info in note_tracks:
        track = mido.MidiTrack()
        
        # Set track name
        track_channels = df[df['track name'] == track_info['name']]['channel'].unique()
        if len(track_channels) > 1:
            track_name = f"{track_info['name']} (Ch {track_info['channel']})"
        else:
            track_name = track_info['name']
            
        track.append(mido.MetaMessage('track_name', name=track_name, time=0))
        
        # Set program
        notes = track_info['notes']
        prog = int(notes['program'].iloc[0])
        track.append(mido.Message('program_change', program=prog, channel=track_info['channel'], time=0))
        
        # Add note events
        events = []
        for _, row in notes.iterrows():
            onset_ticks = int(row['onset in quarter notes'] * target_ticks_per_beat)
            offset_ticks = int((row['onset in quarter notes'] + row['duration in quarter notes']) * target_ticks_per_beat)
            pitch = int(row['pitch'])
            velocity = int(row['velocity'])
            
            events.append((onset_ticks, mido.Message('note_on', note=pitch, velocity=velocity, channel=track_info['channel'])))
            events.append((offset_ticks, mido.Message('note_off', note=pitch, velocity=0, channel=track_info['channel'])))
        
        # Sort and add events with correct delta times
        events.sort(key=lambda x: x[0])
        last_tick = 0
        for abs_tick, msg in events:
            delta = abs_tick - last_tick
            msg.time = delta
            track.append(msg)
            last_tick = abs_tick
        
        mid.tracks.append(track)
    
    # Save the file
    mid.save(output_path)
    print(f"✅ Saved {output_path} with exact timing structure preserved")
    print(f"   - {len(timing_events)} timing events preserved")
    print(f"   - {len(note_tracks)} instrument tracks created")
