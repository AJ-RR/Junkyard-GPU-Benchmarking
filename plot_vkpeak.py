#!/usr/bin/env python3
"""
Visualization script for vkpeak benchmark results.
Plots performance metrics for different GPU devices.
"""

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_vkpeak_file(filepath):
    """Parse vkpeak output file and extract benchmark data."""
    data = {
        'device': None,
        'fp32': {},
        'fp16': {},
        'fp64': {},
        'int32': {},
        'int16': {},
        'int64': {},
        'int8': {},
        'bf16': {},
        'fp8': {},
        'bf8': {},
        'copy': {}
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check if file contains error message
        if 'No such file' in content or 'error' in content.lower():
            return None
            
        # Extract device name
        device_match = re.search(r'device\s*=\s*(.+)', content)
        if device_match:
            data['device'] = device_match.group(1).strip()
        
        # Parse different benchmark types
        patterns = {
            'fp32': (r'fp32-(\w+)\s*=\s*([\d.]+)\s*GFLOPS', 'GFLOPS'),
            'fp16': (r'fp16-(\w+)\s*=\s*([\d.]+)\s*GFLOPS', 'GFLOPS'),
            'fp64': (r'fp64-(\w+)\s*=\s*([\d.]+)\s*GFLOPS', 'GFLOPS'),
            'int32': (r'int32-(\w+)\s*=\s*([\d.]+)\s*GIOPS', 'GIOPS'),
            'int16': (r'int16-(\w+)\s*=\s*([\d.]+)\s*GIOPS', 'GIOPS'),
            'int64': (r'int64-(\w+)\s*=\s*([\d.]+)\s*GIOPS', 'GIOPS'),
            'int8': (r'int8-(\w+)\s*=\s*([\d.]+)\s*GIOPS', 'GIOPS'),
            'bf16': (r'bf16-(\w+)\s*=\s*([\d.]+)\s*GFLOPS', 'GFLOPS'),
            'fp8': (r'fp8-(\w+)\s*=\s*([\d.]+)\s*GFLOPS', 'GFLOPS'),
            'bf8': (r'bf8-(\w+)\s*=\s*([\d.]+)\s*GFLOPS', 'GFLOPS'),
            'copy': (r'copy-(\w+)\s*=\s*([\d.]+)\s*GBPS', 'GBPS'),
        }
        
        for key, (pattern, unit) in patterns.items():
            matches = re.findall(pattern, content)
            for match_type, value in matches:
                data[key][match_type] = float(value)
        
        return data if data['device'] else None
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def plot_fp32_performance(data_dict):
    """Plot FP32 performance metrics."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('FP32 Performance Comparison', fontsize=24, fontweight='bold')
    
    devices = list(data_dict.keys())
    # Use same color scheme as LLM plots
    color_map = {'Mali-G710': '#2E86AB', 'Tesla T4': '#A23B72'}
    colors = [color_map.get(device, '#808080') for device in devices]
    
    fp32_types = ['scalar', 'vec4']
    x = np.arange(len(fp32_types))
    width = 0.35
    
    for i, device in enumerate(devices):
        values = [data_dict[device]['fp32'].get(t, 0) for t in fp32_types]
        ax.bar(x + i*width, values, width, label=device, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Operation Type', fontsize=16)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
    ax.set_xticks(x + width * (len(devices)-1) / 2)
    ax.set_xticklabels(fp32_types, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Set log scale
    
    plt.tight_layout()
    return fig

def plot_integer_performance(data_dict):
    """Plot integer performance metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Integer Performance Comparison', fontsize=24, fontweight='bold')
    
    devices = list(data_dict.keys())
    # Use same color scheme as LLM plots
    color_map = {'Mali-G710': '#2E86AB', 'Tesla T4': '#A23B72'}
    colors = [color_map.get(device, '#808080') for device in devices]
    
    int_types = ['int32-scalar', 'int32-vec4', 'int16-scalar', 'int16-vec4', 
                 'int64-scalar', 'int64-vec4']
    x = np.arange(len(int_types))
    width = 0.35
    
    for i, device in enumerate(devices):
        values = []
        for int_type in int_types:
            dtype, op = int_type.split('-')
            values.append(data_dict[device][dtype].get(op, 0))
        ax.bar(x + i*width, values, width, label=device, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Operation Type', fontsize=16)
    ax.set_ylabel('Performance (GIOPS)', fontsize=16)
    ax.set_xticks(x + width * (len(devices)-1) / 2)
    ax.set_xticklabels([t.replace('-', '\n') for t in int_types], fontsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Set log scale
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_fp32_int32_int64_combined(data_dict):
    """Plot FP32, INT32, and INT64 performance metrics together."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('FP32, INT32, and INT64 Performance Comparison', fontsize=24, fontweight='bold')
    
    devices = list(data_dict.keys())
    # Use same color scheme as LLM plots
    color_map = {'Mali-G710': '#2E86AB', 'Tesla T4': '#A23B72'}
    colors = [color_map.get(device, '#808080') for device in devices]
    
    # Define categories: FP32, INT32, INT64, each with scalar and vec4
    categories = [
        'FP32\nScalar', 'FP32\nVec4',
        'INT32\nScalar', 'INT32\nVec4',
        'INT64\nScalar', 'INT64\nVec4'
    ]
    x = np.arange(len(categories))
    width = 0.35
    
    for i, device in enumerate(devices):
        values = []
        # FP32 values
        values.append(data_dict[device]['fp32'].get('scalar', 0))
        values.append(data_dict[device]['fp32'].get('vec4', 0))
        # INT32 values
        values.append(data_dict[device]['int32'].get('scalar', 0))
        values.append(data_dict[device]['int32'].get('vec4', 0))
        # INT64 values
        values.append(data_dict[device]['int64'].get('scalar', 0))
        values.append(data_dict[device]['int64'].get('vec4', 0))
        
        ax.bar(x + i*width, values, width, label=device, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Operation Type', fontsize=16)
    ax.set_ylabel('Performance (GFLOPS/GIOPS)', fontsize=16)
    ax.set_xticks(x + width * (len(devices)-1) / 2)
    ax.set_xticklabels(categories, fontsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Set log scale
    
    plt.tight_layout()
    return fig

def plot_memory_bandwidth(data_dict):
    """Plot memory bandwidth metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Memory Bandwidth Performance Comparison', fontsize=24, fontweight='bold')
    
    devices = list(data_dict.keys())
    copy_types = ['h2h', 'h2d', 'd2h', 'd2d']
    x = np.arange(len(copy_types))
    width = 0.35
    
    # Use same color scheme as LLM plots
    color_map = {'Mali-G710': '#2E86AB', 'Tesla T4': '#A23B72'}
    colors = [color_map.get(device, '#808080') for device in devices]
    
    for i, device in enumerate(devices):
        values = [data_dict[device]['copy'].get(t, 0) for t in copy_types]
        ax.bar(x + i*width, values, width, label=device, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Memory Transfer Type', fontsize=16)
    ax.set_ylabel('Bandwidth (GBPS)', fontsize=16)
    ax.set_xticks(x + width * (len(devices)-1) / 2)
    ax.set_xticklabels(['Host→Host', 'Host→Device', 'Device→Host', 'Device→Device'], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_summary_radar(data_dict):
    """Create a radar chart comparing key metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    devices = list(data_dict.keys())
    
    # Select key metrics for radar chart
    categories = [
        'FP32 Vec4', 'FP16 Vec4', 'INT32 Vec4', 
        'INT16 Vec4', 'INT8 DotProd', 'Copy D2D'
    ]
    
    # Extract values for each category
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Use same color scheme as LLM plots
    color_map = {'Mali-G710': '#2E86AB', 'Tesla T4': '#A23B72'}
    colors = [color_map.get(device, '#808080') for device in devices]
    
    for i, device in enumerate(devices):
        values = []
        for cat in categories:
            if 'FP32' in cat:
                values.append(data_dict[device]['fp32'].get('vec4', 0))
            elif 'FP16' in cat:
                values.append(data_dict[device]['fp16'].get('vec4', 0))
            elif 'INT32' in cat:
                values.append(data_dict[device]['int32'].get('vec4', 0))
            elif 'INT16' in cat:
                values.append(data_dict[device]['int16'].get('vec4', 0))
            elif 'INT8' in cat:
                values.append(data_dict[device]['int8'].get('dotprod', 0))
            elif 'Copy' in cat:
                values.append(data_dict[device]['copy'].get('d2d', 0))
        
        # Normalize values to 0-100 scale for better visualization
        max_val = max([max(vals) for vals in [[data_dict[d]['fp32'].get('vec4', 0),
                                               data_dict[d]['fp16'].get('vec4', 0),
                                               data_dict[d]['int32'].get('vec4', 0),
                                               data_dict[d]['int16'].get('vec4', 0),
                                               data_dict[d]['int8'].get('dotprod', 0),
                                               data_dict[d]['copy'].get('d2d', 0)] for d in devices]])
        
        if max_val > 0:
            normalized_values = [v / max_val * 100 for v in values]
        else:
            normalized_values = values
        
        normalized_values += normalized_values[:1]  # Complete the circle
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, label=device, color=colors[i])
        ax.fill(angles, normalized_values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Normalized Performance (%)', labelpad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    return fig

def plot_comprehensive_comparison(data_dict):
    """Create a comprehensive comparison with all metrics."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    devices = list(data_dict.keys())
    # Use same color scheme as LLM plots
    color_map = {'Mali-G710': '#2E86AB', 'Tesla T4': '#A23B72'}
    colors = [color_map.get(device, '#808080') for device in devices]
    
    # Collect all metrics for heatmap-style visualization
    metrics = []
    metric_names = []
    
    # Floating point metrics
    for dtype in ['fp32', 'fp16', 'fp64']:
        for op in ['scalar', 'vec4', 'matrix']:
            if op in data_dict[devices[0]][dtype]:
                metric_names.append(f'{dtype.upper()}-{op}')
                metrics.append([data_dict[d][dtype].get(op, 0) for d in devices])
    
    # Integer metrics
    for dtype in ['int32', 'int16', 'int64', 'int8']:
        for op in ['scalar', 'vec4', 'dotprod', 'matrix']:
            if op in data_dict[devices[0]][dtype]:
                metric_names.append(f'{dtype.upper()}-{op}')
                metrics.append([data_dict[d][dtype].get(op, 0) for d in devices])
    
    # Memory metrics
    for op in ['h2h', 'h2d', 'd2h', 'd2d']:
        metric_names.append(f'Copy-{op}')
        metrics.append([data_dict[d]['copy'].get(op, 0) for d in devices])
    
    # Create horizontal bar chart
    ax = fig.add_subplot(gs[:, :])
    y_pos = np.arange(len(metric_names))
    
    x_max = max([max(row) for row in metrics]) if metrics else 1
    
    for i, device in enumerate(devices):
        values = [row[i] for row in metrics]
        ax.barh(y_pos + i*0.8/len(devices), values, 0.8/len(devices), 
                label=device, color=colors[i], alpha=0.8)
    
    ax.set_yticks(y_pos + 0.4)
    ax.set_yticklabels(metric_names, fontsize=8)
    ax.set_xlabel('Performance (GFLOPS/GIOPS/GBPS)', fontsize=12)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    return fig

def plot_mali_vs_tesla_comparison(data_dict):
    """Create a dedicated side-by-side comparison plot for Mali vs Tesla GPUs."""
    # Get data using explicit device names
    mali_data = data_dict.get('Mali-G710', None)
    tesla_data = data_dict.get('Tesla T4', None)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mali-G710 vs Tesla GPU Performance Comparison', fontsize=16, fontweight='bold')
    
    def get_value(data, category, key, default=0):
        """Helper to safely get values from data dict."""
        if data is None:
            return default
        return data[category].get(key, default)
    
    # 1. FP32 Performance
    ax = axes[0, 0]
    categories = ['Scalar', 'Vec4']
    mali_vals = [get_value(mali_data, 'fp32', 'scalar'), get_value(mali_data, 'fp32', 'vec4')]
    tesla_vals = [get_value(tesla_data, 'fp32', 'scalar'), get_value(tesla_data, 'fp32', 'vec4')]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, mali_vals, width, label='Mali-G710', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, tesla_vals, width, label='Tesla', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Operation Type')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. FP16 Performance
    ax = axes[0, 1]
    categories = ['Scalar', 'Vec4', 'Matrix']
    mali_vals = [get_value(mali_data, 'fp16', 'scalar'), 
                 get_value(mali_data, 'fp16', 'vec4'),
                 get_value(mali_data, 'fp16', 'matrix')]
    tesla_vals = [get_value(tesla_data, 'fp16', 'scalar'),
                  get_value(tesla_data, 'fp16', 'vec4'),
                  get_value(tesla_data, 'fp16', 'matrix')]
    x = np.arange(len(categories))
    ax.bar(x - width/2, mali_vals, width, label='Mali-G710', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, tesla_vals, width, label='Tesla', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Operation Type')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Integer Performance (INT32, INT16, INT64)
    ax = axes[0, 2]
    categories = ['INT32\nScalar', 'INT32\nVec4', 'INT16\nScalar', 'INT16\nVec4', 
                  'INT64\nScalar', 'INT64\nVec4']
    mali_vals = [
        get_value(mali_data, 'int32', 'scalar'),
        get_value(mali_data, 'int32', 'vec4'),
        get_value(mali_data, 'int16', 'scalar'),
        get_value(mali_data, 'int16', 'vec4'),
        get_value(mali_data, 'int64', 'scalar'),
        get_value(mali_data, 'int64', 'vec4')
    ]
    tesla_vals = [
        get_value(tesla_data, 'int32', 'scalar'),
        get_value(tesla_data, 'int32', 'vec4'),
        get_value(tesla_data, 'int16', 'scalar'),
        get_value(tesla_data, 'int16', 'vec4'),
        get_value(tesla_data, 'int64', 'scalar'),
        get_value(tesla_data, 'int64', 'vec4')
    ]
    x = np.arange(len(categories))
    ax.bar(x - width/2, mali_vals, width, label='Mali-G710', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, tesla_vals, width, label='Tesla', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Operation Type')
    ax.set_ylabel('Performance (GIOPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. INT8 Performance
    ax = axes[1, 0]
    categories = ['DotProd', 'Matrix']
    mali_vals = [get_value(mali_data, 'int8', 'dotprod'), get_value(mali_data, 'int8', 'matrix')]
    tesla_vals = [get_value(tesla_data, 'int8', 'dotprod'), get_value(tesla_data, 'int8', 'matrix')]
    x = np.arange(len(categories))
    ax.bar(x - width/2, mali_vals, width, label='Mali-G710', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, tesla_vals, width, label='Tesla', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Operation Type')
    ax.set_ylabel('Performance (GIOPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Memory Bandwidth
    ax = axes[1, 1]
    categories = ['H2H', 'H2D', 'D2H', 'D2D']
    mali_vals = [
        get_value(mali_data, 'copy', 'h2h'),
        get_value(mali_data, 'copy', 'h2d'),
        get_value(mali_data, 'copy', 'd2h'),
        get_value(mali_data, 'copy', 'd2d')
    ]
    tesla_vals = [
        get_value(tesla_data, 'copy', 'h2h'),
        get_value(tesla_data, 'copy', 'h2d'),
        get_value(tesla_data, 'copy', 'd2h'),
        get_value(tesla_data, 'copy', 'd2d')
    ]
    x = np.arange(len(categories))
    ax.bar(x - width/2, mali_vals, width, label='Mali-G710', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, tesla_vals, width, label='Tesla', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Transfer Type')
    ax.set_ylabel('Bandwidth (GBPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Summary Comparison (Top metrics)
    ax = axes[1, 2]
    categories = ['FP32\nVec4', 'FP16\nVec4', 'INT8\nDotProd', 'Copy\nD2D']
    mali_vals = [
        get_value(mali_data, 'fp32', 'vec4'),
        get_value(mali_data, 'fp16', 'vec4'),
        get_value(mali_data, 'int8', 'dotprod'),
        get_value(mali_data, 'copy', 'd2d')
    ]
    tesla_vals = [
        get_value(tesla_data, 'fp32', 'vec4'),
        get_value(tesla_data, 'fp16', 'vec4'),
        get_value(tesla_data, 'int8', 'dotprod'),
        get_value(tesla_data, 'copy', 'd2d')
    ]
    x = np.arange(len(categories))
    ax.bar(x - width/2, mali_vals, width, label='Mali-G710', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, tesla_vals, width, label='Tesla', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Key Metrics')
    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add note if data is missing
    if mali_data is None or tesla_data is None:
        missing = []
        if mali_data is None:
            missing.append("Mali-G710")
        if tesla_data is None:
            missing.append("Tesla")
        fig.text(0.5, 0.02, f'Note: Missing data for {", ".join(missing)}. Showing available data only.', 
                ha='center', fontsize=10, style='italic', color='red')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to parse files and generate plots."""
    base_dir = Path(__file__).parent
    
    # Explicitly map file paths to device names
    data_dict = {}
    
    # Mali-G710 data from fold directory
    fold_file = base_dir / 'fold' / 'vkpeak_fold_results.txt'
    if fold_file.exists():
        data = parse_vkpeak_file(fold_file)
        if data:
            # Force device name to Mali-G710
            data['device'] = 'Mali-G710'
            data_dict['Mali-G710'] = data
            print(f"Loaded Mali-G710 data from {fold_file}")
    
    # Tesla data from tesla directory
    tesla_file = base_dir / 'tesla' / 'vkpeak_output.txt'
    if tesla_file.exists():
        data = parse_vkpeak_file(tesla_file)
        if data:
            # Force device name to Tesla T4
            data['device'] = 'Tesla T4'
            data_dict['Tesla T4'] = data
            print(f"Loaded Tesla T4 data from {tesla_file}")
    
    if not data_dict:
        print("No valid vkpeak data found!")
        return
    
    print(f"Found data for devices: {', '.join(data_dict.keys())}")
    
    # Generate plots
    print("Generating plots...")
    
    # FP32 Performance plot
    fig1 = plot_fp32_performance(data_dict)
    fig1.savefig(base_dir / 'fp32_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: fp32_performance.png")
    plt.close(fig1)
    
    # Integer Performance plot
    fig2 = plot_integer_performance(data_dict)
    fig2.savefig(base_dir / 'integer_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: integer_performance.png")
    plt.close(fig2)
    
    # Memory bandwidth plot
    fig3 = plot_memory_bandwidth(data_dict)
    fig3.savefig(base_dir / 'memory_bandwidth.png', dpi=300, bbox_inches='tight')
    print("Saved: memory_bandwidth.png")
    plt.close(fig3)
    
    # Combined FP32, INT32, INT64 plot
    fig4 = plot_fp32_int32_int64_combined(data_dict)
    fig4.savefig(base_dir / 'fp32_int32_int64_combined.png', dpi=300, bbox_inches='tight')
    print("Saved: fp32_int32_int64_combined.png")
    plt.close(fig4)
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()

