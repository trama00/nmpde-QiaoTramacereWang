#!/usr/bin/env python3
"""
Plot dispersion study results from dispersion_summary.csv

This script visualizes phase velocity errors and energy conservation
for different θ values, time steps, mesh sizes, and modes.

Usage:
    python plot_dispersion.py [--csv FILE] [--out DIR]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_file):
    """Load dispersion summary CSV."""
    df = pd.read_csv(csv_file)
    return df


def plot_energy_vs_theta(df, out_dir=None):
    """Plot energy ratio ET/E0 vs theta for different modes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by mode and theta, average over N and dt
    modes = df.groupby(['m', 'n'])
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, ((m, n), group) in enumerate(modes):
        # Average over mesh sizes for each theta
        avg = group.groupby('theta')['ET_over_E0'].mean()
        std = group.groupby('theta')['ET_over_E0'].std()
        
        ax.errorbar(avg.index, avg.values, yerr=std.values,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    label=f'mode ({m},{n})',
                    capsize=3, linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Exact (E=const)')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('$E_T / E_0$', fontsize=12)
    ax.set_title('Energy Conservation vs θ-scheme Parameter', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, 'dispersion_energy_vs_theta.png'), dpi=200)
        print(f"Saved: {os.path.join(out_dir, 'dispersion_energy_vs_theta.png')}")
    else:
        plt.show()
    plt.close()


def plot_energy_vs_dt(df, out_dir=None):
    """Plot energy ratio vs dt for different theta values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thetas = df['theta'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))
    
    for idx, theta in enumerate(sorted(thetas)):
        subset = df[df['theta'] == theta]
        # Average over modes and mesh sizes for each dt
        avg = subset.groupby('dt')['ET_over_E0'].mean()
        std = subset.groupby('dt')['ET_over_E0'].std()
        
        ax.errorbar(avg.index, avg.values, yerr=std.values,
                    marker='o', color=colors[idx],
                    label=f'θ={theta}',
                    capsize=3, linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Exact')
    ax.set_xlabel('Δt', fontsize=12)
    ax.set_ylabel('$E_T / E_0$', fontsize=12)
    ax.set_title('Energy Conservation vs Time Step', fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Finer dt to the right
    
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, 'dispersion_energy_vs_dt.png'), dpi=200)
        print(f"Saved: {os.path.join(out_dir, 'dispersion_energy_vs_dt.png')}")
    else:
        plt.show()
    plt.close()


def plot_energy_vs_cfl(df, out_dir=None):
    """Plot energy ratio vs CFL number."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thetas = df['theta'].unique()
    markers = ['o', 's', '^']
    
    for idx, theta in enumerate(sorted(thetas)):
        subset = df[df['theta'] == theta]
        ax.scatter(subset['cfl'], subset['ET_over_E0'],
                   marker=markers[idx % len(markers)],
                   alpha=0.7, s=50,
                   label=f'θ={theta}')
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Exact')
    ax.set_xlabel('CFL number (c·Δt/h)', fontsize=12)
    ax.set_ylabel('$E_T / E_0$', fontsize=12)
    ax.set_title('Energy Conservation vs CFL Number', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, 'dispersion_energy_vs_cfl.png'), dpi=200)
        print(f"Saved: {os.path.join(out_dir, 'dispersion_energy_vs_cfl.png')}")
    else:
        plt.show()
    plt.close()


def plot_energy_vs_wavenumber(df, out_dir=None):
    """Plot energy ratio vs wavenumber |k|."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thetas = df['theta'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))
    
    for idx, theta in enumerate(sorted(thetas)):
        subset = df[df['theta'] == theta]
        # Average over dt and N for each wavenumber
        avg = subset.groupby('k_mag')['ET_over_E0'].mean()
        std = subset.groupby('k_mag')['ET_over_E0'].std()
        
        ax.errorbar(avg.index, avg.values, yerr=std.values,
                    marker='o', color=colors[idx],
                    label=f'θ={theta}',
                    capsize=3, linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Exact')
    ax.set_xlabel('Wavenumber $|k| = \\pi\\sqrt{m^2+n^2}$', fontsize=12)
    ax.set_ylabel('$E_T / E_0$', fontsize=12)
    ax.set_title('Energy Conservation vs Wavenumber (Higher modes = more dispersion)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, 'dispersion_energy_vs_k.png'), dpi=200)
        print(f"Saved: {os.path.join(out_dir, 'dispersion_energy_vs_k.png')}")
    else:
        plt.show()
    plt.close()


def plot_phase_error_heatmap(df, out_dir=None):
    """Plot heatmap of phase error proxy for different (h, dt) combinations."""
    # Filter for theta=0.5 (Crank-Nicolson) and mode (1,1)
    subset = df[(df['theta'] == 0.5) & (df['m'] == 1) & (df['n'] == 1)]
    
    if subset.empty:
        print("No data for θ=0.5, mode (1,1) found for heatmap.")
        return
    
    # Create pivot table
    pivot = subset.pivot_table(values='phase_error_proxy', 
                               index='h', columns='dt', 
                               aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.4f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{x:.4f}' for x in pivot.index])
    
    ax.set_xlabel('Δt', fontsize=12)
    ax.set_ylabel('h (mesh spacing)', fontsize=12)
    ax.set_title('Phase Error Proxy |ET/E0 - 1| (θ=0.5, mode (1,1))', fontsize=14)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Phase Error Proxy')
    
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, 'dispersion_heatmap.png'), dpi=200)
        print(f"Saved: {os.path.join(out_dir, 'dispersion_heatmap.png')}")
    else:
        plt.show()
    plt.close()


def plot_summary_table(df, out_dir=None):
    """Print and optionally save a summary table."""
    # Group by theta and compute statistics
    summary = df.groupby('theta').agg({
        'ET_over_E0': ['mean', 'std', 'min', 'max'],
        'phase_error_proxy': ['mean', 'max']
    }).round(6)
    
    print("\n" + "="*60)
    print("DISPERSION STUDY SUMMARY")
    print("="*60)
    print(summary.to_string())
    print("\nNote: For Crank-Nicolson (θ=0.5), energy should be conserved (ET/E0 ≈ 1)")
    print("      For θ > 0.5, numerical dissipation causes ET/E0 < 1")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot dispersion study results from CSV"
    )
    parser.add_argument("--csv", default="results/dispersion/dispersion_summary.csv",
                        help="Input CSV file")
    parser.add_argument("--out", default="",
                        help="Output directory for plots (default: show plots)")
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Run the DispersionStudy executable first:")
        print("  cd build && mpirun -np 1 ./DispersionStudy")
        return
    
    df = load_data(args.csv)
    print(f"Loaded {len(df)} records from {args.csv}")
    
    out_dir = args.out if args.out else None
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Generate all plots
    plot_summary_table(df, out_dir)
    plot_energy_vs_theta(df, out_dir)
    plot_energy_vs_dt(df, out_dir)
    plot_energy_vs_cfl(df, out_dir)
    plot_energy_vs_wavenumber(df, out_dir)


if __name__ == "__main__":
    main()
