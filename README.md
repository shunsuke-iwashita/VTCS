# VTCS - Visual Tactical Context System

A comprehensive Python framework for analyzing tactical movements and scenarios in ultimate frisbee using computer vision and data analysis.

## Overview

VTCS (Visual Tactical Context System) provides tools for analyzing player movements, detecting tactical patterns, and generating scenarios from ultimate frisbee game data. The system processes tracking data to extract meaningful tactical insights and visualize player behaviors.

## Project Structure

The project is organized into focused modules for better maintainability:

### Core Modules

- **`VTCS.py`**: Main interface and orchestration class
- **`data_utils.py`**: Data loading, preprocessing, and utility functions
- **`movement_detector.py`**: Movement pattern detection and analysis
- **`scenario_generator.py`**: Tactical scenario generation and management
- **`evaluator.py`**: Performance evaluation and timing analysis

### Visualization and Tools

- **`VTCS_visualize.py`**: Field visualization and animation generation
- **`draw_bbox.py`**: Bounding box visualization for tracking data

## Features

- **Movement Detection**: Identifies player movements and tactical patterns
- **Scenario Generation**: Creates and shifts tactical scenarios
- **Performance Evaluation**: Analyzes timing and effectiveness metrics
- **Visualization**: Generates field plots and animated sequences
- **Data Processing**: Handles multiple data formats and preprocessing

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from VTCS import VTCS

# Initialize the system
vtcs = VTCS()

# Load and process data
vtcs.load_data(data_path="data/input/")

# Detect movements and generate scenarios
results = vtcs.process_play_data()
```

### Visualization

```python
from VTCS_visualize import plot_play

# Generate animated visualization
plot_play(play_data, save_path="output/animation.mp4")
```

## Data Format

The system expects tracking data with the following structure:
- Player positions (x, y coordinates)
- Frame-based timing information
- Player classifications (offense/defense)
- Movement velocities

## Output

- Tactical scenario files
- Performance evaluation metrics
- Animated visualizations
- Statistical analysis reports

## Development

The codebase follows Google-style docstrings and is organized for modularity and maintainability. Each module has a single responsibility and clear interfaces.

## License

This project is for research and educational purposes in sports analytics.