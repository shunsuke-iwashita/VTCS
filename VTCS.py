import sys

import pandas as pd

import VTCS_visualize as vv
from data_utils import add_derived_columns
from evaluator import VTCSEvaluator
from movement_detector import MovementDetector
from scenario_generator import ScenarioGenerator


class VTCS:
    """Vision-based Tactical Counterfactual Scenarios (VTCS) analysis system.

    A class for analyzing ultimate frisbee player movements and generating
    counterfactual scenarios to evaluate timing decisions.

    Attributes:
        play (pd.DataFrame): Processed tracking data with derived columns.
        movement_detector (MovementDetector): Component for detecting player movements.
        scenario_generator (ScenarioGenerator): Component for generating scenarios.
        evaluator (VTCSEvaluator): Component for evaluating scenarios.
        candidates (Dict[str, pd.DataFrame]): Dictionary of detected movement candidates.
        selected (pd.DataFrame): Currently selected candidate for analysis.
        scenarios (Dict[int, pd.DataFrame]): Generated scenarios with different time shifts.
    """

    def __init__(self, ultimate_track_df):
        """Initialize VTCS with ultimate frisbee tracking data.

        Args:
            ultimate_track_df (pd.DataFrame): DataFrame containing tracking data with columns:
                - 'id': Player/disc ID
                - 'frame': Frame number
                - 'x', 'y': Position coordinates
                - 'vx', 'vy': Velocity components
                - 'ax', 'ay': Acceleration components
                - 'class': Player type ('offense', 'defense', 'disc')
                - 'holder': Boolean indicating if player holds the disc
                - 'closest': ID of closest opposing player
        """
        # Add derived columns to the play data
        self.play = add_derived_columns(ultimate_track_df)

        # Initialize components
        self.movement_detector = MovementDetector(self.play)
        self.scenario_generator = None
        self.evaluator = VTCSEvaluator()

        # Storage for analysis results
        self.candidates = {}
        self.selected = None
        self.scenarios = {}

    def detect_candidates(self):
        """Detect movement candidates from the play DataFrame.

        This method performs a three-step process to identify potential movement
        candidates:

        1. Detect movement initiation based on acceleration and velocity criteria
        2. Deselect movements based on proximity and direction constraints
        3. Extract candidate DataFrames with temporal windows

        The method updates the internal candidates dictionary with detected movements.
        Each candidate is identified by a unique key in the format "{player_id}-{sequence_number}".

        Note:
            This method modifies the internal state of the VTCS object by populating
            the candidates dictionary.
        """
        # Update play data with detection results
        self.movement_detector.detect_movement_initiation()
        self.play = self.movement_detector.play

        # Filter out crowded movements
        self.movement_detector.deselect_crowded_movements()
        self.play = self.movement_detector.play

        # Extract candidates
        self.candidates = self.movement_detector.extract_candidates()

    def generate_scenarios(self, shifts=range(-15, 16)):
        """Generate counterfactual scenarios by shifting selected movement timing.

        Creates alternative scenarios by temporally shifting the selected movement
        either forward (negative shifts) or backward (positive shifts) in time.

        Args:
            shifts (range, optional): Range of temporal shifts to apply in frames.
                Negative values shift movement earlier, positive values shift later.
                Defaults to range(-15, 16) for ±15 frame shifts.

        Raises:
            ValueError: If no candidate has been selected for analysis.

        Note:
            After generating all scenarios, disc positions are automatically
            adjusted to maintain consistency with ball possession.
        """
        if self.selected is None:
            raise ValueError("No candidate selected. Call select_candidate() first.")

        self.scenario_generator = ScenarioGenerator(self.selected)
        self.scenarios = self.scenario_generator.generate_scenarios(shifts)

    def select_candidate(self, candidate_id):
        """Select a candidate for scenario generation.

        Args:
            candidate_id (str): ID of the candidate to select. Must exist in the
                candidates dictionary.

        Raises:
            ValueError: If the specified candidate_id is not found in the candidates
                dictionary.
        """
        if candidate_id not in self.candidates:
            raise ValueError(f"Candidate {candidate_id} not found.")

        self.selected = self.candidates[candidate_id]

    def evaluate(self, file_name: str, candidate_id: str):
        """Evaluate all generated scenarios using frame-wise and scenario-wise metrics.

        Calculates evaluation metrics for each scenario including V_frame (frame-wise
        values), V_scenario (scenario summary), and V_timing (timing effectiveness).

        Args:
            file_name (str): Name of the input file used for evaluation, primarily
                for logging and result identification.
            candidate_id (str): Identifier for the selected candidate movement.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation results with keys:
                - 'v_frame': Dict mapping shift -> list of frame-wise values
                - 'v_scenario': Dict mapping shift -> scenario summary value
                - 'v_timing': Float representing timing effectiveness
                - 'best_timing': Int representing the optimal shift value

        Raises:
            ValueError: If no scenarios have been generated prior to evaluation.

        Note:
            V_timing is calculated as the difference between actual (shift=0)
            and best possible scenario performance. Requires pre-computed wUPPCF
            files to be available in the data/input/player_wUPPCF/ directory.
        """
        if not self.scenarios:
            raise ValueError("No scenarios generated. Call generate_scenarios() first.")

        return self.evaluator.evaluate_scenarios(
            self.scenarios, file_name, candidate_id
        )


def main():
    """Main function to demonstrate VTCS analysis workflow.

    Loads ultimate frisbee tracking data, detects movement candidates,
    allows user selection of a candidate, generates counterfactual scenarios,
    and creates visualizations for analysis.

    The workflow includes:
        1. Load tracking data from CSV
        2. Detect movement candidates
        3. Interactive candidate selection
        4. Generate temporal shift scenarios
        5. Create visualization videos
        6. Evaluate scenario performance

    Example:
        Run the script directly to start the interactive analysis::

            $ python VTCS.py

    Note:
        Requires input data file at 'data/input/UltimateTrack/1_1_2.csv'.
        Evaluation requires corresponding wUPPCF files in 'data/input/player_wUPPCF/'.
    """
    pd.set_option("display.max_rows", None)
    # Load data and initialize VTCS
    file_name = "data/input/UltimateTrack/1_2_1.csv"
    ultimate_track_df = pd.read_csv(file_name)
    vtcs = VTCS(ultimate_track_df)

    # Detect movement candidates
    vtcs.detect_candidates()
    for candidate_id, candidate_df in vtcs.candidates.items():
        vv.plot_play(candidate_df, f"output/{candidate_id}.mp4")

    if vtcs.candidates:
        # Interactive candidate selection
        while True:
            print("Available candidates:", ", ".join(vtcs.candidates.keys()))
            candidate_index = input("Enter candidate index to select (e.g., '1-1'): ")

            if candidate_index == "q":
                print("Exiting.")
                sys.exit(0)

            if candidate_index in vtcs.candidates:
                vtcs.select_candidate(candidate_index)
                if vtcs.selected is not None and hasattr(vtcs.selected, "attrs"):
                    selected_attrs = vtcs.selected.attrs
                    print(
                        f"Selected candidate: {selected_attrs['movement_player_id']} "
                        f"from frame {selected_attrs['movement_start_frame']} "
                        f"to {selected_attrs['movement_end_frame']}"
                    )
                else:
                    print(f"Selected candidate: {candidate_index}")
                break
            else:
                print("Invalid candidate index. Please try again.")

        # Generate scenarios and evaluate
        vtcs.generate_scenarios()

        evaluation_results = vtcs.evaluate(
            file_name.split("/")[-1].split(".")[0], candidate_index
        )

        # Display results
        print("\nEvaluation Results:")
        print("V_frame:", evaluation_results["v_frame"])
        print("V_scenario:", evaluation_results["v_scenario"])
        print("V_timing:", evaluation_results["v_timing"])
        print("Best timing shift:", evaluation_results["best_timing"])


if __name__ == "__main__":
    main()
