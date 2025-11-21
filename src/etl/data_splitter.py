import os
import json
import pickle
import random
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

def is_valid_keypoint(keypoint_file, frame_idx):
    """Check if keypoints at frame_idx are valid (no NaN values)."""
    try:
        with open(keypoint_file, 'rb') as f:
            data = pickle.load(f)
        
        # Get keypoints for this frame: data['keypoints2d'][person_idx][frame_idx]
        # We use person_idx=0 (first person)
        keypoints = data['keypoints2d'][0][frame_idx]  # Shape: (17, 3)
        
        # Check if any NaN values exist in the keypoint data
        return not np.isnan(keypoints).any()
    except Exception as e:
        # If we can't load or access the data, consider it invalid
        return False

def parse_filename(filename):
    attributes = filename.replace('.pkl', '').split('_')

    if len(attributes) != 6:
        return None
    
    return {
        'genre': attributes[0],
        'situation': attributes[1],
        'camera': attributes[2],
        'dancer': attributes[3],
        'music': attributes[4],
        'choreo': attributes[5],
        'filename': filename
    }


def get_file_metadata(keypoint_folder):
    keypoint_files = []
    
    for filename in os.listdir(keypoint_folder):
        if not filename.endswith('.pkl'):
            continue

        metadata = parse_filename(filename)
        if not metadata:
            continue

        filepath = os.path.join(keypoint_folder, filename)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            metadata['num_frames'] = len(data['keypoints2d'][0])
            metadata['path'] = filepath
        
        keypoint_files.append(metadata)

    return keypoint_files

def create_choreo_splits(keypoint_files, train_ratio=0.7, val_ratio = 0.2, seed=42):
    random.seed(seed)

    choreos = sorted(set(f['choreo'] for f in keypoint_files))
    print(f"{len(choreos)} found in keypoint files")

    random.shuffle(choreos)
    n_train = int(len(choreos) * train_ratio)
    n_val = int(len(choreos) * val_ratio)

    train_ch = choreos[:n_train]
    val_ch = choreos[n_train:n_train + n_val]
    test_ch = choreos[n_train + n_val:]

    splits = {
        'train': {'choreographies': train_ch, 'files': []},
        'val': {'choreographies': val_ch, 'files': []},
        'test': {'choreographies': test_ch, 'files': []}
    }


    for metadata in keypoint_files:
        choreo = metadata['choreo']
        if choreo in train_ch:
            splits['train']['files'].append(metadata)
        elif choreo in val_ch:
            splits['val']['files'].append(metadata)
        else:
            splits['test']['files'].append(metadata)

    tot_files = 0
    file_counts = {
        'train': 0,
        'val': 0,
        'test': 0
    }
    for split_name, split_data in splits.items():
        print(f"{split_name}: {len(split_data['files'])} files")
        file_counts[split_name] = len(split_data['files'])
        tot_files += file_counts[split_name]
        
    for split_name in file_counts.keys():
        print(f"{file_counts[split_name] / tot_files:.2%} {split_name} actual ratio")

    return splits

def generate_pairs(split_files, keypoint_folder, num_positive, num_negative, max_frame_diff=30):

    by_choreo = defaultdict(list)

    for metadata in split_files:
        by_choreo[metadata['choreo']].append(metadata)

    pairs = []

    positive_count = 0
    attempts = 0
    max_attempts = num_positive * 10


    # generate positives pairs
    while positive_count < num_positive and attempts < max_attempts:
        attempts += 1

        ch = random.choice(list(by_choreo.keys()))

        if len(by_choreo[ch]) < 2:
            continue

        file1, file2 = random.sample(by_choreo[ch], 2)

        max_frame = min(file1['num_frames'], file2['num_frames']) - 1

        frame_idx = random.randint(0, max_frame)
        
        # Build full paths to check for NaN
        file1_path = os.path.join(keypoint_folder, file1['filename'])
        file2_path = os.path.join(keypoint_folder, file2['filename'])
        
        # Skip this pair if either has NaN values
        if not is_valid_keypoint(file1_path, frame_idx) or not is_valid_keypoint(file2_path, frame_idx):
            continue

        pairs.append({
            'file1': file1['filename'],
            'file2': file2['filename'],
            'frame1': frame_idx,
            'frame2': frame_idx,
            'match': 1,
            'type': 'positive',
            'ch': ch
        })

        positive_count += 1

    difficult_negative_count = 0
    attempts = 0

    while difficult_negative_count < num_negative // 2 and attempts < max_attempts:
        attempts += 1
        ch = random.choice(list(by_choreo.keys()))

        file1, file2 = random.sample(by_choreo[ch], 2)

        if len(by_choreo[ch]) < 2:
            continue

        frame1 = random.randint(0, file1['num_frames'] - 1)
        frame2 = random.randint(0, file2['num_frames'] - 1)

        if abs(frame1 - frame2) < max_frame_diff:
            continue
        
        # Build full paths to check for NaN
        file1_path = os.path.join(keypoint_folder, file1['filename'])
        file2_path = os.path.join(keypoint_folder, file2['filename'])
        
        # Skip this pair if either has NaN values
        if not is_valid_keypoint(file1_path, frame1) or not is_valid_keypoint(file2_path, frame2):
            continue

        pairs.append({
            'file1': file1['filename'],
            'file2': file2['filename'],
            'frame1': frame1,
            'frame2': frame2,
            'match': 0,
            'type': 'difficult',
            'ch': ch
        })
        difficult_negative_count += 1


    easy_negative_count = 0
    attempts = 0

    while easy_negative_count < num_negative // 2 and attempts < max_attempts:
        attempts += 1

        file1 = random.choice(split_files)
        file2 = random.choice(split_files)

        if file1['choreo'] == file2['choreo']:
            continue

        frame1 = random.randint(0, file1['num_frames'] - 1)
        frame2 = random.randint(0, file2['num_frames'] - 1)
        
        # Build full paths to check for NaN
        file1_path = os.path.join(keypoint_folder, file1['filename'])
        file2_path = os.path.join(keypoint_folder, file2['filename'])
        
        # Skip this pair if either has NaN values
        if not is_valid_keypoint(file1_path, frame1) or not is_valid_keypoint(file2_path, frame2):
            continue

        pairs.append({
            'file1': file1['filename'],
            'file2': file2['filename'],
            'frame1': frame1,
            'frame2': frame2,
            'match': 0,
            'type': 'easy',
            'ch': None
        })
        easy_negative_count += 1

    random.shuffle(pairs)

    return pairs

def main():
    keypoint_folder = "../../data/processed/aist_plusplus_final/keypoints2d"
    output_folder = "../../data/raw/splits"

    keypoint_files = get_file_metadata(keypoint_folder)

    splits = create_choreo_splits(keypoint_files, seed=50)

    

    os.makedirs(output_folder, exist_ok=True)

    for split_name, split_data in splits.items():
        output_path = os.path.join(output_folder, f"{split_name}_files.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)

    pair_counts = {
        'train': 10000,
        'val': 2000,
        'test': 2000
    }


    for split_name, split_data in splits.items():
        pair_count = pair_counts[split_name]

        pairs = generate_pairs(split_data['files'], keypoint_folder, pair_count // 2, pair_count // 2)

        output_path = os.path.join(output_folder, f"{split_name}_pairs.json")

        with open(output_path, 'w') as f:
            json.dump(pairs, f, indent=2)

        pos_count = sum(1 for p in pairs if p['match'] == 1)
        hard_neg_count = sum(1 for p in pairs if p['type'] == 'difficult')
        easy_neg_count = sum(1 for p in pairs if p['type'] == 'easy')

        print(f"    Positive pairs: {pos_count}")
        print(f"    Hard negative pairs: {hard_neg_count}")
        print(f"    Easy negative pairs: {easy_neg_count}")


if __name__ == '__main__':
    main()
        

    



    


