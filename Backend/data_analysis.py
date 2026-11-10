import mne
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Phase 1: We will be only considering sleep-casette dataset and build a model using just the EEG dataset and see how it goes 

# Lets start with one single recording !
DIR = Path("Backend/Data/sleep-cassette")
data_filepath = os.path.join(DIR,"SC4001E0-PSG.edf")
label_filepath = os.path.join(DIR,"SC4001EC-Hypnogram.edf")


raw = mne.io.read_raw_edf(data_filepath, preload = True)
annot = mne.read_annotations(label_filepath)
raw.set_annotations(annot)

df = raw.to_data_frame()
df_annot = raw.annotations.to_data_frame()

print("--- Sleep Stage Counts ---")
print(df_annot['description'].value_counts())

stage_durations = df_annot.groupby('description')['duration'].sum() / 3600
print("\n--- Duration per stage (Hours) ---")
print(stage_durations)

events, event_id = mne.events_from_annotations(raw)

epochs = mne.Epochs(raw, events=events, event_id=event_id, 
                    tmin=0, tmax=30.0, baseline=None, preload=True)

print(epochs)