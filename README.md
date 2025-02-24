# A* Path Finding Algorithm Visualization

This project implements an interactive visualization of the A* path-finding algorithm using Python and Pygame. It includes sound effects and a user interface with controls displayed on-screen.

## Requirements
- Python 3 and above
- Pygame (`pip install pygame`)
- NumPy (`pip install numpy`)

## Installation
1. Clone or download this repository
2. Install the required packages
3. Run the script

## Technical Details
- Window size: 800x900 pixels (800x800 grid + 100px for controls)
- Grid: 50x50 cells
- Heuristic: Manhattan distance
- Sound:
  - Search beeps: 200-1000 Hz sine wave, 0.05s duration
  - Path beeps: 600 Hz triangle wave, 0.03s duration
  - Both with fade-in/out envelopes to prevent clicking
