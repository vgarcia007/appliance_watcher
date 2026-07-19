#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile Learner for Appliance Watcher

- Scans all logs in ./logs
- Extracts statistical features from each cycle
- Groups cycles into device-specific profiles (simple clustering)
- Calculates duration statistics for each profile
- Assigns profile names (Profil 1, Profil 2, ...)
- Saves learned profiles to profiles.json

Usage:
    python3 profile_learn.py
"""
import os
import json
import glob
import statistics
from collections import defaultdict

LOG_DIR = "./logs"
PROFILE_OUT = "./logs/profiles.json"
FEATURE_KEYS = [
    'avg_power', 'max_power', 'min_power', 'std_power', 'median_power',
    'high_power_100_ratio', 'high_power_200_ratio', 'high_power_400_ratio', 'variation_coefficient'
]


def extract_features(powers):
    if len(powers) < 10:
        return None
    avg_power = sum(powers) / len(powers)
    max_power = max(powers)
    min_power = min(powers)
    power_std = statistics.stdev(powers) if len(powers) > 1 else 0
    power_median = statistics.median(powers)
    high_power_100 = sum(1 for p in powers if p >= 100) / len(powers)
    high_power_200 = sum(1 for p in powers if p >= 200) / len(powers)
    high_power_400 = sum(1 for p in powers if p >= 400) / len(powers)
    variation_coeff = power_std / avg_power if avg_power > 0 else 0
    return {
        'avg_power': avg_power,
        'max_power': max_power,
        'min_power': min_power,
        'std_power': power_std,
        'median_power': power_median,
        'high_power_100_ratio': high_power_100,
        'high_power_200_ratio': high_power_200,
        'high_power_400_ratio': high_power_400,
        'variation_coefficient': variation_coeff,
    }


def load_cycles():
    cycles = []
    for log_file in glob.glob(os.path.join(LOG_DIR, '*.json')):
        # profiles.json is this script's output, not a recorded appliance cycle.
        if os.path.abspath(log_file) == os.path.abspath(PROFILE_OUT):
            continue
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            powers = [entry['power_w'] for entry in data.get('data', [])]
            features = extract_features(powers)
            if features:
                duration = data.get('duration_s', 0)
                if not isinstance(duration, (int, float)) or duration <= 0:
                    duration = 0
                cycles.append({
                    'file': log_file,
                    'features': features,
                    'duration': duration,
                    'device': data.get('device', 'unknown'),
                    'started_at': data.get('started_at'),
                    'finished_at': data.get('finished_at'),
                })
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    return cycles


def feature_distance(f1, f2):
    dist = 0.0
    count = 0
    for k in FEATURE_KEYS:
        if k in f1 and k in f2:
            scale = max(abs(f1[k]), abs(f2[k]), 1)
            dist += abs(f1[k] - f2[k]) / scale
            count += 1
    return dist / count if count else 1.0


def cluster_cycles(cycles, threshold=0.3):
    profiles = []
    for cycle in cycles:
        best_profile = None
        best_distance = float('inf')

        # Never compare or merge cycles belonging to different appliances.
        for profile in profiles:
            if profile['device'] != cycle['device']:
                continue
            dist = feature_distance(cycle['features'], profile['features'])
            if dist < threshold and dist < best_distance:
                best_profile = profile
                best_distance = dist

        if best_profile is None:
            profiles.append({
                'features': dict(cycle['features']),
                'cycles': [cycle['file']],
                '_durations': [cycle['duration']] if cycle['duration'] > 0 else [],
                '_feature_samples': 1,
                'device': cycle['device'],
            })
            continue

        # Maintain a true arithmetic mean instead of repeatedly averaging the
        # previous center with the newest cycle.
        sample_count = best_profile['_feature_samples']
        for key in FEATURE_KEYS:
            old_value = best_profile['features'][key]
            new_value = cycle['features'][key]
            best_profile['features'][key] = (
                old_value * sample_count + new_value
            ) / (sample_count + 1)
        best_profile['_feature_samples'] = sample_count + 1
        best_profile['cycles'].append(cycle['file'])
        if cycle['duration'] > 0:
            best_profile['_durations'].append(cycle['duration'])

    # Add stable profile names and duration statistics used by the watcher.
    for i, profile in enumerate(profiles):
        profile['name'] = f"Profil {i+1}"
        durations = profile.pop('_durations')
        profile.pop('_feature_samples')
        profile['duration_samples'] = len(durations)
        profile['duration_median_s'] = (
            int(round(statistics.median(durations))) if durations else None
        )
        profile['duration_min_s'] = min(durations) if durations else None
        profile['duration_max_s'] = max(durations) if durations else None
    return profiles


def save_profiles(profiles):
    with open(PROFILE_OUT, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(profiles)} profiles to {PROFILE_OUT}")


def main():
    print("Loading cycles from logs...")
    cycles = load_cycles()
    print(f"Loaded {len(cycles)} cycles.")
    print("Clustering cycles...")
    profiles = cluster_cycles(cycles)
    print(f"Found {len(profiles)} profiles.")
    save_profiles(profiles)

if __name__ == "__main__":
    main()
