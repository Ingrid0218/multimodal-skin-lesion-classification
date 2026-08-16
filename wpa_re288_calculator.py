"""
wpa_matrix_exporter_b1.py
Version B-1: MLB RE288 Game-State Matrix Exporter

Systematically generates all 57,024 baseline game states across innings,
half-innings, outs, base states, balls, strikes, and score differentials.
Computes count-specific Run Expectancy (RE288) and Home Win Expectancy (WE),
exporting the complete matrix to a structured CSV file.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class WPAStateMatrixExporterB1:
    """
    Constructs, validates, and exports the complete 57,024-permutation 
    MLB RE288 win probability matrix.
    """

    def __init__(self, output_filename: str = "re288_baseline_wpa_matrix.csv"):
        self.output_path: Path = Path(output_filename)
        self.base_out_re: Dict[str, Dict[int, float]] = self._init_base_out_re()
        self.ball_weights: Dict[int, float] = {0: 0.000, 1: 0.033, 2: 0.072, 3: 0.145}
        self.strike_weights: Dict[int, float] = {0: 0.000, 1: -0.055, 2: -0.120}

    @staticmethod
    def _init_base_out_re() -> Dict[str, Dict[int, float]]:
        """
        Returns the foundational 24-state Run Expectancy (RE) matrix.
        Mapping: Base State -> Outs -> Expected Runs.
        """
        return {
            "000": {0: 0.481, 1: 0.254, 2: 0.098},
            "100": {0: 0.859, 1: 0.509, 2: 0.224},
            "010": {0: 1.100, 1: 0.664, 2: 0.319},
            "001": {0: 1.350, 1: 0.950, 2: 0.350},
            "110": {0: 1.437, 1: 0.884, 2: 0.429},
            "101": {0: 1.784, 1: 1.130, 2: 0.471},
            "011": {0: 1.964, 1: 1.376, 2: 0.580},
            "111": {0: 2.282, 1: 1.541, 2: 0.752}
        }

    def _calculate_win_expectancy(
        self, 
        score_diff: pd.Series, 
        re_value: pd.Series, 
        half_inning: pd.Series, 
        inning: pd.Series, 
        outs: pd.Series
    ) -> pd.Series:
        """
        Vectorized calculation of Home Team Win Expectancy using a dynamic 
        logistic model scaled by game progression (outs remaining).
        """
        # Top = Away batting (RE hurts Home); Bottom = Home batting (RE helps Home)
        effective_lead = np.where(
            half_inning == 'Top',
            score_diff - re_value,
            score_diff + re_value
        )

        # Standard regulation game = 9 innings = 54 total half-inning outs
        half_inning_offset = np.where(half_inning == 'Top', 0, 3)
        elapsed_outs = ((inning - 1) * 6) + half_inning_offset + outs
        outs_remaining = np.clip(54 - elapsed_outs, 1, 54)

        # Dynamic leverage scaling as outs decrease
        game_progress = (54.0 - outs_remaining) / 54.0
        scale_factor = 0.45 + (1.5 * (game_progress ** 2))

        # Logistic probability calculation
        win_expectancy = 1.0 / (1.0 + np.exp(-scale_factor * effective_lead))
        return pd.Series(np.round(win_expectancy, 4), index=score_diff.index)

    def generate_matrix(self) -> pd.DataFrame:
        """
        Executes the Cartesian product of all 7 game state dimensions, maps
        and vectorizes RE288 count adjustments, and computes Win Expectancy.
        """
        # Define discrete dimension spaces
        innings: List[int] = list(range(1, 10))
        half_innings: List[str] = ['Top', 'Bottom']
        outs: List[int] = [0, 1, 2]
        base_states: List[str] = ['000', '100', '010', '001', '110', '101', '011', '111']
        balls: List[int] = [0, 1, 2, 3]
        strikes: List[int] = [0, 1, 2]
        score_diffs: List[int] = list(range(-5, 6))

        # Build Cartesian product efficiently using MultiIndex
        index = pd.MultiIndex.from_product(
            [innings, half_innings, outs, base_states, balls, strikes, score_diffs],
            names=['Inning', 'Half_Inning', 'Outs', 'Base_State', 'Balls', 'Strikes', 'Home_Score_Diff']
        )
        
        df = index.to_frame().reset_index(drop=True)

        # Assert architectural validity
        expected_rows = (
            len(innings) * len(half_innings) * len(outs) * len(base_states) * len(balls) * len(strikes) * len(score_diffs)
        )
        assert len(df) == expected_rows, f"Matrix size mismatch! Expected {expected_rows}, got {len(df)}"

        # 1. Vectorized Base-Out Run Expectancy mapping
        re_lookup_df = pd.DataFrame([
            {"Base_State": b_state, "Outs": o_val, "Base_RE": re_val}
            for b_state, out_map in self.base_out_re.items()
            for o_val, re_val in out_map.items()
        ])
        df = df.merge(re_lookup_df, on=['Base_State', 'Outs'], how='left')

        # 2. Vectorized Ball/Strike Count Adjustments (RE288 Engineering)
        ball_shift = df['Balls'].map(self.ball_weights)
        strike_shift = df['Strikes'].map(self.strike_weights)
        
        re_val = df['Base_RE'] + ball_shift + strike_shift
        
        # Apply leverage extreme conditions (3-0 boost, 0-2 penalty)
        re_val = np.where((df['Balls'] == 3) & (df['Strikes'] == 0), re_val + 0.050, re_val)
        re_val = np.where((df['Balls'] == 0) & (df['Strikes'] == 2), re_val - 0.040, re_val)
        
        # Clamp RE so it cannot fall below minimum threshold
        df['RE_Value'] = np.round(np.maximum(0.010, re_val), 4)

        # 3. Calculate Count-Specific Baseline Win Expectancy
        df['Baseline_Win_Expectancy'] = self._calculate_win_expectancy(
            score_diff=df['Home_Score_Diff'],
            re_value=df['RE_Value'],
            half_inning=df['Half_Inning'],
            inning=df['Inning'],
            outs=df['Outs']
        )

        # Drop intermediate calculation columns and enforce explicit order
        final_columns = [
            'Inning', 
            'Half_Inning', 
            'Outs', 
            'Base_State', 
            'Balls',
            'Strikes',
            'Home_Score_Diff', 
            'RE_Value',
            'Baseline_Win_Expectancy'
        ]
        
        return df[final_columns]

    def export_to_csv(self) -> Tuple[pd.DataFrame, Path]:
        """
        Generates the 57,024-row RE288 matrix and writes it safely to disk.
        Returns the DataFrame and the resolved file path.
        """
        print(f"Generating 57,024-permutation RE288 baseline WPA matrix...")
        matrix_df = self.generate_matrix()
        
        # Safe writing using pathlib
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(self.output_path, index=False, encoding='utf-8')
        
        print(f"Successfully exported {len(matrix_df):,} rows to: {self.output_path.resolve()}")
        return matrix_df, self.output_path


if __name__ == "__main__":
    exporter = WPAStateMatrixExporterB1(output_filename="re288_baseline_wpa_matrix.csv")
    df_matrix, file_path = exporter.export_to_csv()
    
    # Display terminal verification summary
    print("\n--- MATRIX VERIFICATION SUMMARY ---")
    print(f"Total Permutations Generated: {len(df_matrix):,}")
    print(f"Null Values Present:          {df_matrix.isnull().sum().sum()}")
    print("\nFirst 5 Rows:")
    print(df_matrix.head(5).to_string(index=False))