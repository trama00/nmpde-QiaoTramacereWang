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
steps = list(range(0, 101, 10))
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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top plot: displacement
line1, = ax1.plot([], [], 'b-', linewidth=2, label='Displacement')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.2, 1.2)
ax1.set_xlabel('Position x', fontsize=12)
ax1.set_ylabel('Displacement u(x,t)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
title = ax1.set_title('', fontsize=14, fontweight='bold')

# Bottom plot: all snapshots overlaid
ax2.set_xlabel('Position x', fontsize=12)
ax2.set_ylabel('Displacement u(x,t)', fontsize=12)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-0.2, 1.2)
ax2.grid(True, alpha=0.3)
ax2.set_title('All Time Steps Overlaid', fontsize=12)

# Plot all time steps
colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
for i, (step, d) in enumerate(data.items()):
    time = step * 0.01
    ax2.plot(d[:, 0], d[:, 1], color=colors[i], alpha=0.6, 
             linewidth=1, label=f't={time:.1f}' if step % 50 == 0 else '')

ax2.legend(fontsize=10)

plt.tight_layout()

def init():
    line1.set_data([], [])
    return line1,

def animate(frame):
    step = list(data.keys())[frame]
    d = data[step]
    line1.set_data(d[:, 0], d[:, 1])
    time = step * 0.01
    title.set_text(f'1D Wave Equation: Step {step}, Time = {time:.2f} s, Energy = 0.576')
    return line1,

anim = FuncAnimation(fig, animate, init_func=init,
                    frames=len(data), interval=200, blit=False, repeat=True)

print("\nDisplaying animation. Close the window when done.")
print("(The animation will loop continuously)")

plt.show()

# Optionally save the animation
save = input("\nSave animation as GIF? (y/n): ").lower().strip()
if save == 'y':
    print("Saving animation (this may take a moment)...")
    anim.save('wave_animation.gif', writer='pillow', fps=5)
    print("Saved as wave_animation.gif")
