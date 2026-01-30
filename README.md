# 3-Body Orbit Explorer

I created this Python application to visualize the 3-Body problem using various stable periodic orbits. It uses Matplotlib for the 3D visualization and SciPy for the physics calculations.

## Features

* **3D Visualization:** View the movement of three bodies in a fully interactive 3D plot.
* **Speed Control:** A slider allows you to speed up the simulation (by skipping calculation frames) or slow it down for precision.
* **Visualization Settings:** Options to change trail length (short, long, infinite) and object size during the simulation.
* **Presets:** Includes known stable orbits like the Figure-8 (Chenciner), Lagrange points, and the Sitnikov problem.
* **Data Export:** Save position and velocity data for all bodies to a CSV file.

## Dependencies

This project requires the following Python libraries:
* `numpy`
* `scipy`
* `matplotlib`
* `pandas`

## Installation

1. Ensure you have Python installed.
2. Install the required libraries:

```bash
pip install numpy pandas scipy matplotlib
