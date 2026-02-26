"""
visualize_contact_points.py

================================================================================
CONTACT POINT COLLISION ANALYSIS VISUALIZATION
================================================================================

PURPOSE:
  This script analyzes and visualizes collision detection results with focus
  on contact points between objects. It generates statistical reports and
  charts to understand collision patterns without requiring video files.

FUNCTIONALITY:
  - Loads collision events and object trajectory data
  - Analyzes contact point statistics (front, center, back)
  - Generates comprehensive statistical plots:
    * Distance distribution histogram
    * Time-to-Collision (TTC) distribution
    * Contact point type frequency
    * Temporal distribution of near-miss events
  - Creates summary statistics by object pair
  - Identifies high-risk collision scenarios
  - Lightweight analysis (no video processing required)

USAGE:
  Full analysis with plots:
    python visualize_contact_points.py \
      --near-misses results/xxx/5_collision_analysis/collision_events.json \
      --tracks results/xxx/5_collision_analysis/tracks.json \
      --output analysis_results/

  Data analysis only (no visualization):
    python visualize_contact_points.py \
      --near-misses collision_events.json \
      --tracks tracks.json \
      --analyze-only

PARAMETERS:
  --near-misses <path>     : Path to collision events JSON file (REQUIRED)
  --tracks <path>          : Path to object trajectories JSON file (REQUIRED)
  --output <path>          : Output directory for plots and reports
                             (Default: visualization/)
  --analyze-only           : Only perform analysis, skip plot generation
                             (useful for quick data inspection)

OUTPUT:
  Files generated:
    - contact_points_analysis.png  : 4-subplot analysis chart
    - Console output               : Detailed statistics and rankings

  Chart contents (4 subplots):
    1. Distance Distribution      : Histogram of all collision distances
    2. TTC Distribution           : Histogram of Time-to-Collision values
    3. Contact Point Types        : Bar chart of contact point combinations
    4. Temporal Distribution      : Scatter plot showing events over time

ANALYSIS FEATURES:
  - Contact point type statistics (front-front, center-back, etc.)
  - High-risk events filtering (TTC < 3 seconds)
  - Top closest contact points ranking
  - Object pair interaction analysis
  - Event frequency distribution

OUTPUT STATISTICS:
  Contact Point Types:
    F = Front (leading edge of vehicle)
    C = Center (center point of object)
    B = Back (trailing edge of vehicle)

  Risk Classification:
    - Very High: TTC < 0.5s
    - High:      TTC < 2.0s
    - Medium:    TTC 2.0-3.0s
    - Low:       TTC > 3.0s

ADVANTAGES:
  ✓ Fast execution (no video processing)
  ✓ Comprehensive statistical analysis
  ✓ Helps identify collision patterns
  ✓ Useful for algorithm validation
  ✓ Generates publication-ready plots

================================================================================
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import cv2
from pathlib import Path


def analyze_contact_points(near_misses_path, tracks_path):
    """
    Analyze contact point collision data and print comprehensive statistics
    
    Parameters:
      near_misses_path: Path to collision events JSON
      tracks_path: Path to object trajectories JSON
    """
    
    print("=" * 60)
    print("Contact Point Analysis")
    print("=" * 60)
    
    # Load data
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)
    
    print(f"\n✓ Loaded {len(near_misses)} near-miss events")
    print(f"✓ Loaded {len(tracks)} tracked objects\n")
    
    if not near_misses:
        print("⚠ No near-miss events found")
        return
    
    # Contact point type statistics
    print("【Contact Point Statistics】")
    print("-" * 60)
    
    point_type_stats = defaultdict(int)
    for event in near_misses:
        if 'closest_point_pair' in event:
            point_pair = event['closest_point_pair']
            key = f"{point_pair['obj1_point_type']}-{point_pair['obj2_point_type']}"
            point_type_stats[key] += 1
    
    if point_type_stats:
        print("Collision point type distribution:")
        for point_pair, count in sorted(point_type_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {point_pair}: {count} events")
    else:
        print("  No contact point data (using fallback center distance)")
    
    # High-risk collision analysis
    print("\n【High-Risk Events (TTC < 3s)】")
    print("-" * 60)
    
    collision_risks = [nm for nm in near_misses if nm.get('is_collision_risk', False)]
    print(f"Total high-risk events: {len(collision_risks)}")
    
    if collision_risks:
        # Sort by distance
        sorted_by_distance = sorted(collision_risks, key=lambda x: x['distance'])
        print("\nTop 5 closest contact points:")
        for i, event in enumerate(sorted_by_distance[:5], 1):
            obj1, obj2 = event['id1'], event['id2']
            dist = event['distance']
            ttc = event['ttc']
            
            if 'closest_point_pair' in event:
                point_info = event['closest_point_pair']
                pt_type = f"{point_info['obj1_point_type']}-{point_info['obj2_point_type']}"
                print(f"\n  {i}. Object {obj1} vs Object {obj2}")
                print(f"     Contact: {pt_type} ({dist:.2f} units)")
                print(f"     TTC: {ttc:.2f}s (if TTC < 0.5s, very critical!)")
            else:
                print(f"\n  {i}. Object {obj1} vs Object {obj2}")
                print(f"     Distance: {dist:.2f} units")
                print(f"     TTC: {ttc:.2f}s")
    
    # Object pair statistics
    print("\n【Object Pair Statistics】")
    print("-" * 60)
    
    pair_stats = defaultdict(list)
    for event in near_misses:
        key = (min(event['id1'], event['id2']), max(event['id1'], event['id2']))
        pair_stats[key].append(event)
    
    print(f"Total unique object pairs: {len(pair_stats)}")
    print("\nPairs with most near-miss events:")
    for pair, events in sorted(pair_stats.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        obj1, obj2 = pair
        min_dist = min(e['distance'] for e in events)
        min_ttc = min((e['ttc'] for e in events if e['ttc'] is not None), default=None)
        
        print(f"  Object {obj1} ↔ Object {obj2}: {len(events)} events, min_distance={min_dist:.2f}, min_ttc={min_ttc}")


def create_summary_plot(near_misses_path, output_dir):
    """
    Generate comprehensive statistical plots from collision data
    
    Parameters:
      near_misses_path: Path to collision events JSON
      output_dir: Directory where plots will be saved
    """
    
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    
    if not near_misses:
        print("No data to plot")
        return
    
    # Prepare data
    distances = [nm['distance'] for nm in near_misses]
    ttcs = [nm['ttc'] for nm in near_misses if nm['ttc'] is not None]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Distance Distribution
    axes[0, 0].hist(distances, bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distance Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Distance (units)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(sum(distances)/len(distances), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Subplot 2: TTC Distribution
    if ttcs:
        axes[0, 1].hist(ttcs, bins=30, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('TTC Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time to Collision (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(2.0, color='red', linestyle='--', label='Warning threshold')
        axes[0, 1].legend()
    
    # Subplot 3: Contact Point Type Distribution
    point_type_stats = defaultdict(int)
    for event in near_misses:
        if 'closest_point_pair' in event:
            point_pair = event['closest_point_pair']
            key = f"{point_pair['obj1_point_type'][0].upper()}-{point_pair['obj2_point_type'][0].upper()}"
            point_type_stats[key] += 1
    
    if point_type_stats:
        axes[1, 0].bar(point_type_stats.keys(), point_type_stats.values(), color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Contact Point Types', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Contact Type (F=Front, C=Center, B=Back)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 4: Temporal Distribution
    timestamps = [nm['timestamp'] for nm in near_misses]
    axes[1, 1].scatter(timestamps, distances, alpha=0.6, s=30, color='purple')
    axes[1, 1].set_title('Near-miss Events Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('Distance (units)')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'contact_points_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Analysis plot saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize contact point collision analysis')
    parser.add_argument('--near-misses', required=True, help='Path to near_misses.json')
    parser.add_argument('--tracks', required=True, help='Path to tracks.json')
    parser.add_argument('--output', default='visualization', help='Output directory')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze data, do not create video')
    
    args = parser.parse_args()
    
    # 数据分析
    analyze_contact_points(args.near_misses, args.tracks)
    
    # 生成图表
    print("\n【Generating Summary Plot】")
    print("-" * 60)
    create_summary_plot(args.near_misses, args.output)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output}")
