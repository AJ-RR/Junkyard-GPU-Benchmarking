#!/usr/bin/env python3
"""
Visualization script for comparing Gemma LLM performance between Mali G710 and Tesla GPUs.
Parses llama.cpp benchmark output and creates comparison plots.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_llamacpp_benchmark(filepath):
    """Parse llama.cpp benchmark output file and extract performance data."""
    data = {
        'device': None,
        'model_size': None,
        'tests': {}
    }
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Extract device name from ggml_vulkan line
        for line in lines:
            if 'ggml_vulkan:' in line and 'Found' in line:
                # Look for device name in subsequent lines
                continue
            elif 'ggml_vulkan:' in line and '=' in line:
                # Extract device name: "ggml_vulkan: 0 = Tesla T4 (NVIDIA) | ..."
                match = re.search(r'=\s*([^(]+)', line)
                if match:
                    device_name = match.group(1).strip()
                    # Clean up device name
                    if 'Mali' in device_name:
                        data['device'] = 'Mali-G710'
                    elif 'Tesla' in device_name:
                        data['device'] = 'Tesla T4'
                    else:
                        data['device'] = device_name
                break
        
        # Parse benchmark table
        # Format: | model | size | params | backend | ngl | test | t/s |
        in_table = False
        for line in lines:
            # Check if we're in the table section
            if '|' in line and 'test' in line.lower() and 't/s' in line.lower():
                in_table = True
                continue
            
            if in_table and '|' in line and 'gemma' in line.lower():
                # Parse the benchmark line
                # Example: | gemma3 1B Q4_K - Medium | 762.49 MiB | 999.89 M | Vulkan | 99 | pp512 | 6287.85 ± 1259.79 |
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 7:
                    # Extract model size if not already set
                    if data['model_size'] is None and len(parts) >= 3:
                        model_size = parts[2].strip()  # size column
                        data['model_size'] = model_size
                    
                    test_name = parts[6].strip()  # test column
                    t_s_value = parts[7].strip()  # t/s column
                    
                    # Extract the mean value (before ±)
                    match = re.search(r'([\d.]+)', t_s_value)
                    if match:
                        mean_value = float(match.group(1))
                        # Extract uncertainty if present
                        uncertainty_match = re.search(r'±\s*([\d.]+)', t_s_value)
                        uncertainty = float(uncertainty_match.group(1)) if uncertainty_match else 0.0
                        
                        data['tests'][test_name] = {
                            'mean': mean_value,
                            'uncertainty': uncertainty
                        }
        
        return data if data['device'] and data['tests'] else None
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def plot_gemma_comparison(mali_data, tesla_data):
    """Create comparison plot for Gemma LLM performance."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Build title with model size (only once since both use same model)
    title = 'Gemma LLM Performance'
    model_size = None
    if mali_data and mali_data.get('model_size'):
        model_size = mali_data['model_size']
    elif tesla_data and tesla_data.get('model_size'):
        model_size = tesla_data['model_size']
    
    if model_size:
        title += f" ({model_size})"
    
    fig.suptitle(title, fontsize=24, fontweight='bold', y=1.02)
    
    # Collect test names from both datasets
    all_tests = set()
    if mali_data and mali_data['tests']:
        all_tests.update(mali_data['tests'].keys())
    if tesla_data and tesla_data['tests']:
        all_tests.update(tesla_data['tests'].keys())
    
    all_tests = sorted(list(all_tests))
    
    if not all_tests:
        # No data available
        fig.text(0.5, 0.5, 'No benchmark data available.\nPlease ensure both data files contain valid benchmark results.',
                ha='center', va='center', fontsize=12, style='italic')
        plt.tight_layout()
        return fig
    
    # Bar chart comparison
    x = np.arange(len(all_tests))
    width = 0.35
    
    mali_means = []
    mali_uncertainties = []
    tesla_means = []
    tesla_uncertainties = []
    
    for test in all_tests:
        if mali_data and test in mali_data['tests']:
            mali_means.append(mali_data['tests'][test]['mean'])
            mali_uncertainties.append(mali_data['tests'][test]['uncertainty'])
        else:
            mali_means.append(0)
            mali_uncertainties.append(0)
        
        if tesla_data and test in tesla_data['tests']:
            tesla_means.append(tesla_data['tests'][test]['mean'])
            tesla_uncertainties.append(tesla_data['tests'][test]['uncertainty'])
        else:
            tesla_means.append(0)
            tesla_uncertainties.append(0)
    
    bars1 = ax.bar(x - width/2, mali_means, width, label='Mali-G710', 
                    color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, tesla_means, width, label='Tesla T4', 
                    color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Test Type', fontsize=16)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_tests, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visualization
    
    # Add value labels above bars with clear spacing
    for i, (mali_val, tesla_val) in enumerate(zip(mali_means, tesla_means)):
        if mali_val > 0:
            # Position text above the bar (1x for log scale spacing)
            ax.text(i - width/2, mali_val * 1, f'{mali_val:.1f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        if tesla_val > 0:
            # Position text above the bar (1x for log scale spacing)
            ax.text(i + width/2, tesla_val * 1, f'{tesla_val:.1f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add note if data is missing
    notes = []
    if not mali_data or not mali_data['tests']:
        notes.append("Mali-G710 data is missing or empty")
    if not tesla_data or not tesla_data['tests']:
        notes.append("Tesla T4 data is missing or empty")
    
    if notes:
        fig.text(0.5, 0.02, f'Note: {"; ".join(notes)}', 
                ha='center', fontsize=10, style='italic', color='orange')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to parse files and generate comparison plot."""
    base_dir = Path(__file__).parent
    
    # Parse Mali GPU data
    mali_file = base_dir / 'fold' / 'llama_gemma31b_fold.txt'
    mali_data = None
    if mali_file.exists():
        mali_data = parse_llamacpp_benchmark(mali_file)
        if mali_data:
            print(f"Loaded Mali-G710 data from {mali_file}")
            print(f"  Device: {mali_data['device']}")
            print(f"  Tests: {list(mali_data['tests'].keys())}")
        else:
            print(f"Warning: Could not parse Mali-G710 data from {mali_file}")
            print("  File may be empty or in unexpected format")
    else:
        print(f"Warning: Mali-G710 data file not found: {mali_file}")
    
    # Parse Tesla GPU data
    tesla_file = base_dir / 'tesla' / 'llamacpp_bench.txt'
    tesla_data = None
    if tesla_file.exists():
        tesla_data = parse_llamacpp_benchmark(tesla_file)
        if tesla_data:
            print(f"\nLoaded Tesla T4 data from {tesla_file}")
            print(f"  Device: {tesla_data['device']}")
            print(f"  Tests: {list(tesla_data['tests'].keys())}")
        else:
            print(f"Warning: Could not parse Tesla T4 data from {tesla_file}")
    else:
        print(f"Warning: Tesla T4 data file not found: {tesla_file}")
    
    if not mali_data and not tesla_data:
        print("\nError: No valid benchmark data found!")
        return
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    fig = plot_gemma_comparison(mali_data, tesla_data)
    
    # Save plot
    output_file = base_dir / 'gemma_performance_comparison.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    print("\nPlot generated successfully!")
    plt.show()

if __name__ == '__main__':
    main()

