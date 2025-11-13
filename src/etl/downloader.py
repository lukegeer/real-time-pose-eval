# coding=utf-8
# Copyright 2020 The Google AI Perception Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download AIST++ videos from AIST Dance Video Database website.

Be aware: Before running this script to download the videos, you should have read
the Terms of Use of the AIST Dance Video Database here:

https://aistdancedb.ongaaccel.jp/terms_of_use/
"""
import argparse
import multiprocessing
import os
import sys
import urllib.request
from functools import partial

SOURCE_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'
LIST_URL = 'https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt'

def _download(video_url, download_folder):
  save_path = os.path.join(download_folder, os.path.basename(video_url))
  urllib.request.urlretrieve(video_url, save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Scripts for downloading AIST++ videos.')
  parser.add_argument(
      '--download_folder',
      type=str,
      required=True,
      help='where to store AIST++ videos.')
  parser.add_argument(
      '--num_processes',
      type=int,
      default=1,
      help='number of threads for multiprocessing.')
  args = parser.parse_args()

  ans = input(
      "Before running this script, please make sure you have read the <Terms of Use> "
      "of AIST Dance Video Database at here: \n"
      "\n"
      "https://aistdancedb.ongaaccel.jp/terms_of_use/\n"
      "\n"
      "Do you agree with the <Terms of Use>? [Y/N]"
  )
  if ans in ["Yes", "YES", "yes", "Y", "y"]:
    pass
  else:
    print ("Program exit. Please first acknowledge the <Terms of Use>.")
    exit()

  os.makedirs(args.download_folder, exist_ok=True)

  seq_names = urllib.request.urlopen(LIST_URL)
  seq_names = [seq_name.strip().decode('utf-8') for seq_name in seq_names]
  
  # Filter for camera c01 only
  seq_names = [name for name in seq_names if "_c01_" in name]
  print(f"Found {len(seq_names)} videos for camera c01")
  
  # Filter for sBM only
  seq_names = [name for name in seq_names if "sBM" in name]
  seq_names = seq_names[:10]

  # Define all genres
  genres = ['gBR', 'gPO', 'gLO', 'gMH', 'gLH', 'gHO', 'gWA', 'gKR', 'gJS', 'gJB']
  
  # Process each genre
  for genre in genres:
    # Filter videos by genre
    genre_videos = [name for name in seq_names if name.startswith(genre + "_")]
    
    if not genre_videos:
      print(f"No videos found for genre {genre}, skipping...")
      continue
    
    print(f"\n{'='*60}")
    print(f"Processing genre: {genre} ({len(genre_videos)} videos)")
    print(f"{'='*60}")
    
    # Process each choreography within this genre
    for ch in range(11):
      ch_str = str(ch).zfill(2)
      
      # Filter by choreography
      ch_videos = [name for name in genre_videos if f"_ch{ch_str}" in name]
      
      if not ch_videos:
        continue
      
      # Create folder structure: download_folder/genre/ch##/
      download_folder = os.path.join(args.download_folder, genre, f"ch{ch_str}")
      os.makedirs(download_folder, exist_ok=True)
      
      # Create video URLs
      video_urls = [os.path.join(SOURCE_URL, seq_name + '.mp4') for seq_name in ch_videos]
      
      print(f"\n  Downloading {len(video_urls)} videos for {genre}/ch{ch_str}...")
      
      download_func = partial(_download, download_folder=download_folder)
      pool = multiprocessing.Pool(processes=args.num_processes)
      for i, _ in enumerate(pool.imap_unordered(download_func, video_urls)):
        sys.stderr.write(f'\r  [{genre}/ch{ch_str}] downloading %d / %d' % (i + 1, len(video_urls)))
      sys.stderr.write('\n')
    
    print(f"Completed genre {genre}")
  
  print(f"\n{'='*60}")
  print("All downloads complete!")
  print(f"{'='*60}")
