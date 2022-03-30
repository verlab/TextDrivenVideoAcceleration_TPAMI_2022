#!/bin/bash

## Download VaTeX Videos (We used the kinetics-datasets-downloader tool to download the available videos from YouTube)
# NOTE: VaTeX is composed of the VALIDATION split of the Kinetics-600 dataset; therefore, you must modify the script to download the validation videos only. 
# We adpated the function download_test_set in the kinetics-datasets-downloader/downloader/download.py file to do so.

# 1. Clone repository and copy the modified files
git clone https://github.com/dancelogue/kinetics-datasets-downloader/ VaTeX_downloader_files/kinetics-datasets-downloader/
cp VaTeX_downloader_files/download.py VaTeX_downloader_files/kinetics-datasets-downloader/downloader/download.py
cp VaTeX_downloader_files/config.py VaTeX_downloader_files/kinetics-datasets-downloader/downloader/lib/config.py

# 2. Get the kinetics dataset annotations
wget -O kinetics600.tar.gz https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz
tar -xf kinetics600.tar.gz -C VaTeX_downloader_files/
rm kinetics600.tar.gz

# 3. Download the videos (This can take a while (~28k videos to download)... If you want, you can stop it at any time and train with the downloaded videos)
python3 VaTeX_downloader_files/kinetics-datasets-downloader/downloader/download.py --val

# Troubleshooting: If the download stops for a long time, experiment increasing the queue size in the parallel downloader (semantic_encoding/resources/VaTeX_downloader_files/kinetics-datasets-downloader/downloader/lib/parallel_download.py)