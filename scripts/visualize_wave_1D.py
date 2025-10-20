#!/usr/bin/env python3
"""
Visualize 1D Wave Equation Results
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Read all time steps
steps = list(range(0, 601, 1))
data = {}

print("Loading data files...")
for step in steps:
    filename = f'output_step_{step}.txt'
    if not os.path.exists(filename):
        print(f"Warning: Could not find {filename}")
        continue
    try:
        d = np.loadtxt(filename, comments='#')
        data[step] = d
        print(f"  Loaded step {step} (t={step*0.01:.2f})")
    except Exception as e:
        print(f"  Error loading {filename}: {e}")

if not data:
    print("No data files found! Make sure you're in the build directory.")
    sys.exit(1)

# Create animation
print("\nCreating animation...")
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Top plot: displacement
line1, = ax1.plot([], [], 'b-', linewidth=2, label='Displacement')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.2, 4.0)
ax1.set_xlabel('Position x', fontsize=12)
ax1.set_ylabel('Displacement u(x,t)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
title = ax1.set_title('', fontsize=14, fontweight='bold')


plt.tight_layout()

# Prepare frames (sorted) and cache arrays for fast access
steps_sorted = sorted(data.keys())
frames_data = [data[s] for s in steps_sorted]
n_frames = len(frames_data)

# If x coordinates are identical across frames, set them once and only update y
x0 = frames_data[0][:, 0]
x_fixed = all(np.allclose(fd[:, 0], x0) for fd in frames_data)

if x_fixed:
    line1.set_data(x0, frames_data[0][:, 1])

def init():
    if x_fixed:
        line1.set_ydata(frames_data[0][:, 1])
    else:
        line1.set_data([], [])
    return (line1,)

def animate(i):
    d = frames_data[i]
    if x_fixed:
        line1.set_ydata(d[:, 1])
    else:
        line1.set_data(d[:, 0], d[:, 1])
    time = steps_sorted[i] * 0.01
    title.set_text(f'1D Wave Equation: Step {steps_sorted[i]}, Time = {time:.2f} s')
    return (line1,)

# Faster playback: shorter interval and use blitting
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=n_frames, interval=50, blit=True, repeat=True)

print("\nDisplaying animation. Close the window when done.")
print("(The animation will loop continuously)")

plt.show()

# Optionally save the animation
# save = input("\nSave animation as GIF? (y/n): ").lower().strip()
# if save == 'y':
#     print("Saving animation (this may take a moment)...")
#     anim.save('wave_animation.gif', writer='pillow', fps=5)
#     print("Saved as wave_animation.gif")
