import mne
import os
from pathlib import Path

# Phase 1: We will be only considering sleep-casette dataset and build a model using just the EEG dataset and see how it goes 

# Lets start with one single recording !
DIR = Path("Backend/Data/sleep-cassette")
data_filepath = os.path.join(DIR,"SC4001E0-PSG.edf")
label_filepath = os.path.join(DIR,"SC4001EC-Hypnogram.edf")


raw = mne.io.read_raw_edf(data_filepath, preload = True)
annot = mne.read_annotations(label_filepath)
raw.set_annotations(annot)
print(raw.ch_names)
print(raw.info)


print(raw.annotations)

events, event_id = mne.events_from_annotations(raw)
print(events)

raw.plot(start = 60, duration= 30)