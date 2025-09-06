import pandas as pd  # For data manipulation
import numpy as np  # For numerical calculations (RMS)
import logging  # For logging
from sklearn.metrics import mean_squared_error, accuracy_score  # For evaluation metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SportsPredictorEvaluator:
    """
    Evaluates the performance of sports prediction models.
    """

    def __init__(self, predictions_filepath, results_filepath, sport_type='football'):
        """
        Initializes the evaluator with prediction and results data.

        Args:
            predictions_filepath (str): Path to the predictions CSV file.
            results_filepath (str): Path to the results CSV file.
            sport_type (str): 'football' or 'f1'.
        """
        self.predictions = self._load_data(predictions_filepath, 'predictions')
        self.results = self._load_data(results_filepath, 'results')
        self.sport_type = sport_type
        self.comparison_data = self._merge_data()

    def _load_data(self, filepath, name):
        """
        Loads data from a CSV file and logs the process.
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {name} data from {filepath}.")
            return df
        except FileNotFoundError:
            logger.error(f"Error: {filepath} not found.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {name} data from {filepath}: {e}")
            return pd.DataFrame()

    def _merge_data(self):
        """
        Merges predictions and results based on 'EventID'.
        """
        if self.predictions.empty or self.results.empty:
            logger.warning("Cannot merge: one or both dataframes are empty.")
            return pd.DataFrame()

        merged_df = pd.merge(self.predictions, self.results, on='EventID', how='inner')
        if merged_df.empty:
            logger.warning("No matching EventIDs found between predictions and results.")
        else:
            logger.info(f"Merged {len(merged_df)} predictions and results.")
        return merged_df

    def calculate_metrics(self):
        """
        Calculates various metrics for prediction goodness.
        """
        if self.comparison_data.empty:
            return {}

        metrics = {}
        for predictor in self.comparison_data['Predictor'].unique():
            predictor_data = self.comparison_data[self.comparison_data['Predictor'] == predictor]
            metrics[predictor] = self._calculate_predictor_metrics(predictor_data)

        metrics['RandomGuess'] = self._calculate_random_metrics()
        return metrics

    def _calculate_predictor_metrics(self, data):
        """
        Calculates metrics for a specific predictor.
        """
        if self.sport_type == 'football':
            correct_predictions = (data['PredictedOutcome'] == data['ActualOutcome']).sum()
            total_predictions = len(data)
            accuracy = correct_predictions / total_predictions
            return {
                'Accuracy': accuracy,
                'Exact Matches': correct_predictions
            }
        elif self.sport_type == 'f1':
            # For F1, we'll compare finishing positions.
            # Convert comma-separated string to lists of integers.
            data['PredictedPositions'] = data['PredictedOutcome'].apply(lambda x: list(map(int, x.split(','))))
            data['ActualPositions'] = data['ActualOutcome'].apply(lambda x: list(map(int, x.split(','))))

            # Root Mean Squared Error (RMS) for position prediction
            # We need to ensure lists have the same length for RMSE
            rms_errors = []
            for _, row in data.iterrows():
                if len(row['PredictedPositions']) == len(row['ActualPositions']):
                    rms_errors.append(np.sqrt(mean_squared_error(row['PredictedPositions'], row['ActualPositions'])))
                else:
                    logger.warning(f"Skipping EventID {row['EventID']} for RMS: Mismatched prediction/actual length.")
            average_rms = np.mean(rms_errors) if rms_errors else np.nan

            # Exact matching of all positions (strict)
            exact_matches = (data['PredictedOutcome'] == data['ActualOutcome']).sum()
            total_predictions = len(data)
            exact_match_accuracy = exact_matches / total_predictions

            return {
                'Average RMSE': average_rms,
                'Exact Order Match Accuracy': exact_match_accuracy
            }
        else:
            logger.warning(f"Unsupported sport type: {self.sport_type}. No metrics calculated.")
            return {}

    def _calculate_random_metrics(self):
        """
        Calculates baseline metrics for random guessing.
        This provides a comparison point for the actual predictors.
        """
        if self.sport_type == 'football':
            possible_outcomes = ['HomeWin', 'AwayWin', 'Draw']
            num_outcomes = len(possible_outcomes)
            # Random guess accuracy is 1/number of outcomes (assuming equal probability)
            random_accuracy = 1 / num_outcomes
            return {
                'Accuracy': random_accuracy,
                'Exact Matches': len(self.comparison_data) * random_accuracy
            }
        elif self.sport_type == 'f1':
            # For F1, a random baseline is more complex and depends on the number of drivers.
            # A simple baseline could be assuming a random permutation of drivers, but RMSE becomes tricky.
            # For simplicity, we can simulate a random exact match accuracy or leave it blank.
            # Here, we'll assume a very low chance of getting the *exact* order right by chance.
            # This is a simplification; a more robust random baseline would require more context.
            num_drivers_per_race = self.results['ActualOutcome'].apply(lambda x: len(x.split(','))).median()
            if not np.isnan(num_drivers_per_race):
                # Probability of guessing the exact ordered sequence of N items is 1/N!
                random_exact_match_accuracy = 1 / np.math.factorial(int(num_drivers_per_race)) if num_drivers_per_race > 0 else 0
            else:
                random_exact_match_accuracy = 0
            return {
                'Average RMSE': np.nan,  # Hard to define meaningful RMSE for random permutations without specific simulation
                'Exact Order Match Accuracy': random_exact_match_accuracy
            }
        else:
            return {}

    def generate_html_report(self, output_filename='prediction_report.html'):
        """
        Generates an HTML report of the evaluation.
        """
        metrics = self.calculate_metrics()
        if not metrics:
            html_content = "<h1>Prediction Evaluation Report</h1><p>No data available for reporting.</p>"
            logger.warning("No metrics to report, skipping HTML generation.")
            with open(output_filename, 'w') as f:
                f.write(html_content)
            return

        html_tables = []
        for predictor, predictor_metrics in metrics.items():
            df_metrics = pd.DataFrame([predictor_metrics])
            df_metrics = df_metrics.rename(index={0: predictor})
            html_tables.append(f"<h2>Metrics for {predictor}</h2>")
            html_tables.append(df_metrics.to_html(classes='table table-striped', border=1))
            html_tables.append("<br>") # Add spacing between tables

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sports Prediction Evaluation Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 80%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .table-striped tbody tr:nth-of-type(odd) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Sports Prediction Evaluation Report ({self.sport_type.capitalize()})</h1>
            {"".join(html_tables)}
        </body>
        </html>
        """

        try:
            with open(output_filename, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML report generated successfully at {output_filename}.")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")

# --- Example Usage ---

# 1. Create dummy CSV files for Football
# predictions_football.csv
# EventID,PredictedOutcome,Predictor
# 1,HomeWin,ModelA
# 2,Draw,ModelA
# 3,AwayWin,ModelB
# 4,HomeWin,ModelB
# 5,Draw,ModelA
# 6,AwayWin,ModelC
#
# results_football.csv
# EventID,ActualOutcome
# 1,HomeWin
# 2,HomeWin
# 3,AwayWin
# 4,Draw
# 5,Draw
# 6,HomeWin

# 2. Create dummy CSV files for F1
# predictions_f1.csv
# EventID,PredictedOutcome,Predictor
# 101,"1,2,3,4,5",ModelX
# 102,"1,3,2,4,5",ModelX
# 103,"2,1,3,4,5",ModelY
# 104,"1,2,3,4,5",ModelY
#
# results_f1.csv
# EventID,ActualOutcome
# 101,"1,2,3,4,5"
# 102,"1,2,3,4,5"
# 103,"1,2,3,4,5"
# 104,"2,1,3,4,5"


# Create dummy files for demonstration
with open("predictions_football.csv", "w") as f:
    f.write("EventID,PredictedOutcome,Predictor\n")
    f.write("1,HomeWin,ModelA\n")
    f.write("2,Draw,ModelA\n")
    f.write("3,AwayWin,ModelB\n")
    f.write("4,HomeWin,ModelB\n")
    f.write("5,Draw,ModelA\n")
    f.write("6,AwayWin,ModelC\n")

with open("results_football.csv", "w") as f:
    f.write("EventID,ActualOutcome\n")
    f.write("1,HomeWin\n")
    f.write("2,HomeWin\n")
    f.write("3,AwayWin\n")
    f.write("4,Draw\n")
    f.write("5,Draw\n")
    f.write("6,HomeWin\n")

with open("predictions_f1.csv", "w") as f:
    f.write("EventID,PredictedOutcome,Predictor\n")
    f.write("101,\"1,2,3,4,5\",ModelX\n")
    f.write("102,\"1,3,2,4,5\",ModelX\n")
    f.write("103,\"2,1,3,4,5\",ModelY\n")
    f.write("104,\"1,2,3,4,5\",ModelY\n")

with open("results_f1.csv", "w") as f:
    f.write("EventID,ActualOutcome\n")
    f.write("101,\"1,2,3,4,5\"\n")
    f.write("102,\"1,2,3,4,5\"\n")
    f.write("103,\"1,2,3,4,5\"\n")
    f.write("104,\"2,1,3,4,5\"\n")


# Evaluate Football Predictions
football_evaluator = SportsPredictorEvaluator("predictions_football.csv", "results_football.csv", sport_type='football')
football_evaluator.generate_html_report("football_prediction_report.html")

# Evaluate F1 Predictions
f1_evaluator = SportsPredictorEvaluator("predictions_f1.csv", "results_f1.csv", sport_type='f1')
f1_evaluator.generate_html_report("f1_prediction_report.html")

