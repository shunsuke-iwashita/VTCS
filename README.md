# VTCS - Valuing Timing by Counterfactual Scenarios

A comprehensive Python framework for analyzing ultimate frisbee player movements and generating counterfactual scenarios to evaluate timing decisions.

## Publication

This work was accepted at the **12th Workshop on Machine Learning and Data Mining for Sports Analytics** (MLSA 2025) at ECML/PKDD 2025 Workshop, Porto, Portugal.

- **Workshop**: [MLSA 2025 Schedule](https://dtai.cs.kuleuven.be/events/MLSA25/schedule.php)
- **arXiv**: [https://arxiv.org/abs/2508.17611](https://arxiv.org/abs/2508.17611)

## Overview

VTCS (Valuing Timing by Counterfactual Scenarios) provides a complete system for analyzing player movements in ultimate frisbee games. The framework detects significant movement patterns, generates counterfactual scenarios by shifting movement timing, and evaluates the strategic value of different timing decisions using pre-computed wUPPCF (weighted Ultimate Probabilistic Player Control Fields) data.

## Key Features

- **Movement Detection**: Automatically identifies significant player movements based on velocity and acceleration criteria
- **Scenario Generation**: Creates counterfactual scenarios by temporally shifting movement timing (±15 frames)
- **Strategic Evaluation**: Analyzes timing effectiveness using frame-wise and scenario-wise metrics
- **Interactive Interface**: Command-line interface for candidate selection and analysis
- **Integrated Pipeline**: All-in-one system combining detection, generation, and evaluation

## Project Structure

### Core Components

- **`vtcs.py`**: Main VTCS system with integrated analysis pipeline
- **`VTCS.py`**: Legacy main interface (deprecated)

### Supporting Modules

- **`VTCS_visualize.py`**: Visualization tools for analysis results
- **`data_utils.py`**: Data processing utilities
- **`movement_detector.py`**: Movement pattern detection
- **`scenario_generator.py`**: Scenario generation
- **`evaluator.py`**: Performance evaluation
- **`draw_bbox.py`**: Bounding box visualization for tracking data

### External Dependencies (Required for wUPPCF calculation)

- **Metrica Sports libraries**: `Metrica_IO.py`, `Metrica_PitchControl.py`, `Metrica_Velocities.py` 
  - These are external libraries required for pitch control calculations
  - Must be installed separately or included in the project directory

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the required data format (see Data Format section below)

## Usage

### Basic Analysis

Run the main VTCS analysis:

```bash
python vtcs.py
```

The system will:
1. Load tracking data from `data/input/UltimateTrack/1_1_2.csv`
2. Detect movement candidates
3. Present an interactive selection interface
4. Generate counterfactual scenarios (±15 frame shifts)
5. Evaluate scenarios using pre-computed wUPPCF files
6. Display evaluation results including V_timing metrics

### Interactive Workflow

```
Available candidates: 1-1, 1-2, 1-3, 3-1, 3-2, 3-3, 3-4, 4-1, 4-2, 4-3, 7-1, 7-2, 10-1, 10-2, 10-3, 11-1, 13-1
Enter candidate index to select (e.g., '1-1'): 1-1
Selected candidate: 1 from frame 36 to 67

Evaluation Results:
V_scenario: {-15: 0.209, -14: 0.212, -13: 0.217, -12: 0.223, -11: 0.229, -10: 0.236, -9: 0.242, -8: 0.245, -7: 0.245, -6: 0.245, -5: 0.25, -4: 0.258, -3: 0.268, -2: 0.281, -1: 0.301, 0: 0.317, 1: 0.339, 2: 0.352, 3: 0.37, 4: 0.39, 5: 0.41, 6: 0.43, 7: 0.447, 8: 0.461, 9: 0.47, 10: 0.47, 11: 0.466, 12: 0.46, 13: 0.455, 14: 0.431, 15: 0.418}
V_timing: -0.15299999999999997
Best timing shift: 9
```

### Programmatic Usage

```python
import pandas as pd
from vtcs import VTCS

# Load data
data = pd.read_csv("your_data.csv")
vtcs = VTCS(data)

# Detect movement candidates
vtcs.detect_candidates()

# Select and analyze a candidate
vtcs.select_candidate("1-1")
vtcs.generate_scenarios()
results = vtcs.evaluate("data_file", "1-1")

print(f"V_timing: {results['v_timing']}")
print(f"Best timing: {results['best_timing']}")
print(f"V_scenario: {results['v_scenario']}")
```

## Data Format

The system expects CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Player/disc identifier |
| `frame` | int | Frame number (temporal sequence) |
| `class` | str | Object type ('offense', 'defense', 'disc') |
| `x`, `y` | float | Position coordinates (meters) |
| `vx`, `vy` | float | Velocity components (m/s) |
| `ax`, `ay` | float | Acceleration components (m/s²) |
| `holder` | bool | Whether player holds the disc |
| `closest` | int | ID of closest opposing player |

## Algorithm Overview

### 1. Movement Detection
- Analyzes acceleration and velocity patterns
- Identifies movement initiation based on configurable thresholds
- Filters out crowded or insignificant movements
- Extracts temporal windows around detected movements

### 2. Scenario Generation
- Creates ±15 frame temporal shifts of selected movements
- Maintains physical consistency through position interpolation
- Resolves disc possession conflicts
- Generates complete scenario data for each shift

### 3. Strategic Evaluation
- Loads pre-computed wUPPCF files from `data/input/player_wUPPCF/`
- Computes V_frame (frame-wise strategic values)
- Calculates V_scenario (scenario summary metrics)
- Determines V_timing (timing effectiveness measure)
- Identifies optimal timing shifts

## Output Metrics

- **V_scenario**: Summary strategic value for each timing shift
- **V_timing**: Difference between actual and optimal timing
- **Best_timing**: Optimal frame shift for maximum strategic value

## Requirements

- Pre-computed wUPPCF files in `data/input/player_wUPPCF/` directory
- Files should be named: `{file_name}-{candidate_id}_{shift}.npy`
- Input tracking data in the specified CSV format

## Configuration

Default parameters can be modified in the source code:

- **Movement thresholds**: `v_threshold=3.0`, `a_threshold=4.0`
- **Temporal shifts**: `range(-15, 16)`
- **Context window**: `window=30` frames
- **Distance threshold**: `distance_threshold=5.0` meters

## Development

The framework uses a modular architecture with clear separation of concerns:
- Movement detection and candidate extraction
- Scenario generation with temporal shifting  
- Strategic evaluation using pre-computed wUPPCF data

All modules follow comprehensive docstring standards and include type hints for better maintainability.

## License

This project is for research and educational purposes in sports analytics and ultimate frisbee tactical analysis.
