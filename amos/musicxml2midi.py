#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Prompt
#Gemini
#Flash 2.5

#A Gamini's response first used music21, but the script did not reconstruct some parts, and warned: "musicxml.m21ToXml: WARNING: we are out of midi channels! help!"
#The following prompt works, after including MuseScore executable. In Mac, get the path by placing the executable in Terminal.

#Then type: nano ~/.bash_profile

#Add the following line to the end of the file, replacing the path with your MuseScore path:
#export PATH="/Applications/MuseScore 4.app/Contents/MacOS:$PATH"

#Save and close the file.

#Apply the changes by running: source ~/.zshrc (or source ~/.bash_profile).

#Prompt: create a python function to convert a MIDI file into MusicXML file, and to convert a Music XML file into a MIDI file. Use a implementation from https://github.com/musescore/MuseScore
#Answer

import subprocess
import os

# Ensure the MuseScore executable is in your system's PATH, or provide the full path to the executable here.
# For example, on Windows, it might be:
# MUSESCORE_EXECUTABLE = "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe"
# On macOS/Linux, it might be:
# MUSESCORE_EXECUTABLE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
# If the executable is in your PATH, you can use just the name like "mscore" or "MuseScore4.exe".
#MUSESCORE_EXECUTABLE = "mscore" # Set this to the correct path if not in your system's PATH
MUSESCORE_EXECUTABLE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
#MUSESCORE_EXECUTABLE = "/Applications/MuseScore 4/Contents/MacOS/mscore"

#Example of use
#midi_file='midis/sugar-plum-fairy_orch.mid'
#output_xml='midis/sugar-plum-fairy_orchB.xml'
#output_file='midis/sugar-plum-fairy_orch_xmlB.mid'
#convert_midi_to_musicxml(midi_file, output_xml)
#convert_musicxml_to_midi(output_xml, output_file)

def convert_midi_to_musicxml(input_midi_file, output_musicxml_file):
    """
    Converts a MIDI file to a MusicXML file using the MuseScore command-line tool.

    Args:
        input_midi_file (str): The path to the input MIDI file.
        output_musicxml_file (str): The path for the output MusicXML file.

    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    if not os.path.exists(input_midi_file):
        print(f"Error: Input MIDI file not found at '{input_midi_file}'")
        return False

    command = [
        MUSESCORE_EXECUTABLE,
        "-o",
        output_musicxml_file,
        input_midi_file
    ]

    try:
        # The capture_output=True and text=True are important for handling output
        # and checking for errors from the subprocess.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Conversion successful. Output saved to '{output_musicxml_file}'")
        print("MuseScore output:")
        print(result.stdout)
        return True
    except FileNotFoundError:
        print(f"Error: MuseScore executable '{MUSESCORE_EXECUTABLE}' not found.")
        print("Please ensure MuseScore is installed and its executable is in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print("MuseScore stderr:")
        print(e.stderr)
        return False

def convert_musicxml_to_midi(input_musicxml_file, output_midi_file):
    """
    Converts a MusicXML file to a MIDI file using the MuseScore command-line tool.

    Args:
        input_musicxml_file (str): The path to the input MusicXML file.
        output_midi_file (str): The path for the output MIDI file.

    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    if not os.path.exists(input_musicxml_file):
        print(f"Error: Input MusicXML file not found at '{input_musicxml_file}'")
        return False

    command = [
        MUSESCORE_EXECUTABLE,
        "-o",
        output_midi_file,
        input_musicxml_file
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Conversion successful. Output saved to '{output_midi_file}'")
        print("MuseScore output:")
        print(result.stdout)
        return True
    except FileNotFoundError:
        print(f"Error: MuseScore executable '{MUSESCORE_EXECUTABLE}' not found.")
        print("Please ensure MuseScore is installed and its executable is in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print("MuseScore stderr:")
        print(e.stderr)
        return False

