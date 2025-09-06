import pandas as pd
import numpy as np
import logging
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import random
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionMetrics:
    """Data class to hold prediction metrics"""
    rms_error: float
    mean_absolute_error: float
    exact_matches: int
    total_predictions: int
    accuracy_percentage: float
    correlation_coefficient: float

class BaseSportsAnalyzer(ABC):
    """Abstract base class for sports prediction analyzers"""
    
    def __init__(self):
        self.predictions_df = None
        self.results_df = None
        self.merged_data = None
        
    @abstractmethod
    def load_predictions(self, filepath: str) -> pd.DataFrame:
        """Load predictions from CSV file"""
        pass
    
    @abstractmethod
    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load actual results from CSV file"""
        pass
    
    @abstractmethod
    def merge_data(self) -> pd.DataFrame:
        """Merge predictions and results data"""
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> Dict[str, PredictionMetrics]:
        """Calculate prediction accuracy metrics"""
        pass
    
    @abstractmethod
    def generate_random_predictions(self) -> pd.DataFrame:
        """Generate random predictions for comparison"""
        pass

class FootballAnalyzer(BaseSportsAnalyzer):
    """Analyzer for football match predictions"""
    
    def __init__(self):
        super().__init__()
        logger.info("Initialized Football Analyzer")
    
    def load_predictions(self, filepath: str) -> pd.DataFrame:
        """Load football predictions from CSV"""
        try:
            logger.info(f"Loading football predictions from {filepath}")
            df = pd.read_csv(filepath)
            
            # Expected columns: match_id, team_a, team_b, predicted_score_a, predicted_score_b, predicted_result
            required_cols = ['match_id', 'team_a', 'team_b', 'predicted_score_a', 'predicted_score_b']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add predicted result if not present
            if 'predicted_result' not in df.columns:
                df['predicted_result'] = df.apply(
                    lambda x: 'win' if x['predicted_score_a'] > x['predicted_score_b'] 
                    else 'loss' if x['predicted_score_a'] < x['predicted_score_b'] 
                    else 'draw', axis=1
                )
            
            self.predictions_df = df
            logger.info(f"Loaded {len(df)} football predictions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading football predictions: {e}")
            raise
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load football results from CSV"""
        try:
            logger.info(f"Loading football results from {filepath}")
            df = pd.read_csv(filepath)
            
            # Expected columns: match_id, team_a, team_b, actual_score_a, actual_score_b, actual_result
            required_cols = ['match_id', 'actual_score_a', 'actual_score_b']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add actual result if not present
            if 'actual_result' not in df.columns:
                df['actual_result'] = df.apply(
                    lambda x: 'win' if x['actual_score_a'] > x['actual_score_b'] 
                    else 'loss' if x['actual_score_a'] < x['actual_score_b'] 
                    else 'draw', axis=1
                )
            
            self.results_df = df
            logger.info(f"Loaded {len(df)} football results")
            return df
            
        except Exception as e:
            logger.error(f"Error loading football results: {e}")
            raise
    
    def merge_data(self) -> pd.DataFrame:
        """Merge predictions and results data"""
        if self.predictions_df is None or self.results_df is None:
            raise ValueError("Both predictions and results must be loaded first")
        
        logger.info("Merging football predictions and results")
        merged = pd.merge(self.predictions_df, self.results_df, on='match_id', how='inner')
        self.merged_data = merged
        logger.info(f"Merged {len(merged)} records")
        return merged
    
    def calculate_metrics(self) -> Dict[str, PredictionMetrics]:
        """Calculate various prediction metrics for football"""
        if self.merged_data is None:
            self.merge_data()
        
        logger.info("Calculating football prediction metrics")
        metrics = {}
        
        # Score prediction metrics
        score_a_rms = np.sqrt(np.mean((self.merged_data['predicted_score_a'] - self.merged_data['actual_score_a'])**2))
        score_b_rms = np.sqrt(np.mean((self.merged_data['predicted_score_b'] - self.merged_data['actual_score_b'])**2))
        combined_rms = np.sqrt((score_a_rms**2 + score_b_rms**2) / 2)
        
        score_a_mae = np.mean(np.abs(self.merged_data['predicted_score_a'] - self.merged_data['actual_score_a']))
        score_b_mae = np.mean(np.abs(self.merged_data['predicted_score_b'] - self.merged_data['actual_score_b']))
        combined_mae = (score_a_mae + score_b_mae) / 2
        
        # Exact score matches
        exact_scores = len(self.merged_data[
            (self.merged_data['predicted_score_a'] == self.merged_data['actual_score_a']) &
            (self.merged_data['predicted_score_b'] == self.merged_data['actual_score_b'])
        ])
        
        # Result prediction accuracy
        correct_results = len(self.merged_data[
            self.merged_data['predicted_result'] == self.merged_data['actual_result']
        ])
        
        total = len(self.merged_data)
        
        # Correlation coefficients
        corr_a = np.corrcoef(self.merged_data['predicted_score_a'], self.merged_data['actual_score_a'])[0,1]
        corr_b = np.corrcoef(self.merged_data['predicted_score_b'], self.merged_data['actual_score_b'])[0,1]
        avg_corr = (corr_a + corr_b) / 2 if not (np.isnan(corr_a) or np.isnan(corr_b)) else 0
        
        metrics['score_prediction'] = PredictionMetrics(
            rms_error=combined_rms,
            mean_absolute_error=combined_mae,
            exact_matches=exact_scores,
            total_predictions=total,
            accuracy_percentage=(exact_scores / total) * 100,
            correlation_coefficient=avg_corr
        )
        
        metrics['result_prediction'] = PredictionMetrics(
            rms_error=0,  # Not applicable for categorical
            mean_absolute_error=0,  # Not applicable for categorical
            exact_matches=correct_results,
            total_predictions=total,
            accuracy_percentage=(correct_results / total) * 100,
            correlation_coefficient=0  # Not applicable for categorical
        )
        
        logger.info(f"Score prediction accuracy: {(exact_scores/total)*100:.2f}%")
        logger.info(f"Result prediction accuracy: {(correct_results/total)*100:.2f}%")
        
        return metrics
    
    def generate_random_predictions(self) -> pd.DataFrame:
        """Generate random football predictions for comparison"""
        if self.results_df is None:
            raise ValueError("Results must be loaded first to generate random predictions")
        
        logger.info("Generating random football predictions")
        random_df = self.results_df.copy()
        
        # Generate random scores (0-5 goals each team)
        random_df['predicted_score_a'] = [random.randint(0, 5) for _ in range(len(random_df))]
        random_df['predicted_score_b'] = [random.randint(0, 5) for _ in range(len(random_df))]
        
        # Generate random results
        random_df['predicted_result'] = random_df.apply(
            lambda x: 'win' if x['predicted_score_a'] > x['predicted_score_b'] 
            else 'loss' if x['predicted_score_a'] < x['predicted_score_b'] 
            else 'draw', axis=1
        )
        
        return random_df

class F1Analyzer(BaseSportsAnalyzer):
    """Analyzer for F1 race predictions"""
    
    def __init__(self):
        super().__init__()
        logger.info("Initialized F1 Analyzer")
    
    def load_predictions(self, filepath: str) -> pd.DataFrame:
        """Load F1 predictions from CSV"""
        try:
            logger.info(f"Loading F1 predictions from {filepath}")
            df = pd.read_csv(filepath)
            
            # Expected columns: race_id, driver, predicted_position, predicted_points
            required_cols = ['race_id', 'driver', 'predicted_position']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add predicted points if not present (F1 2024 points system)
            if 'predicted_points' not in df.columns:
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                df['predicted_points'] = df['predicted_position'].map(lambda x: points_map.get(x, 0))
            
            self.predictions_df = df
            logger.info(f"Loaded {len(df)} F1 predictions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading F1 predictions: {e}")
            raise
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load F1 results from CSV"""
        try:
            logger.info(f"Loading F1 results from {filepath}")
            df = pd.read_csv(filepath)
            
            # Expected columns: race_id, driver, actual_position, actual_points
            required_cols = ['race_id', 'driver', 'actual_position']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add actual points if not present
            if 'actual_points' not in df.columns:
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                df['actual_points'] = df['actual_position'].map(lambda x: points_map.get(x, 0))
            
            self.results_df = df
            logger.info(f"Loaded {len(df)} F1 results")
            return df
            
        except Exception as e:
            logger.error(f"Error loading F1 results: {e}")
            raise
    
    def merge_data(self) -> pd.DataFrame:
        """Merge predictions and results data"""
        if self.predictions_df is None or self.results_df is None:
            raise ValueError("Both predictions and results must be loaded first")
        
        logger.info("Merging F1 predictions and results")
        merged = pd.merge(self.predictions_df, self.results_df, on=['race_id', 'driver'], how='inner')
        self.merged_data = merged
        logger.info(f"Merged {len(merged)} records")
        return merged
    
    def calculate_metrics(self) -> Dict[str, PredictionMetrics]:
        """Calculate various prediction metrics for F1"""
        if self.merged_data is None:
            self.merge_data()
        
        logger.info("Calculating F1 prediction metrics")
        metrics = {}
        
        # Position prediction metrics
        position_rms = np.sqrt(np.mean((self.merged_data['predicted_position'] - self.merged_data['actual_position'])**2))
        position_mae = np.mean(np.abs(self.merged_data['predicted_position'] - self.merged_data['actual_position']))
        exact_positions = len(self.merged_data[
            self.merged_data['predicted_position'] == self.merged_data['actual_position']
        ])
        
        # Points prediction metrics
        points_rms = np.sqrt(np.mean((self.merged_data['predicted_points'] - self.merged_data['actual_points'])**2))
        points_mae = np.mean(np.abs(self.merged_data['predicted_points'] - self.merged_data['actual_points']))
        exact_points = len(self.merged_data[
            self.merged_data['predicted_points'] == self.merged_data['actual_points']
        ])
        
        total = len(self.merged_data)
        
        # Correlation coefficients
        pos_corr = np.corrcoef(self.merged_data['predicted_position'], self.merged_data['actual_position'])[0,1]
        points_corr = np.corrcoef(self.merged_data['predicted_points'], self.merged_data['actual_points'])[0,1]
        
        metrics['position_prediction'] = PredictionMetrics(
            rms_error=position_rms,
            mean_absolute_error=position_mae,
            exact_matches=exact_positions,
            total_predictions=total,
            accuracy_percentage=(exact_positions / total) * 100,
            correlation_coefficient=pos_corr if not np.isnan(pos_corr) else 0
        )
        
        metrics['points_prediction'] = PredictionMetrics(
            rms_error=points_rms,
            mean_absolute_error=points_mae,
            exact_matches=exact_points,
            total_predictions=total,
            accuracy_percentage=(exact_points / total) * 100,
            correlation_coefficient=points_corr if not np.isnan(points_corr) else 0
        )
        
        logger.info(f"Position prediction accuracy: {(exact_positions/total)*100:.2f}%")
        logger.info(f"Points prediction accuracy: {(exact_points/total)*100:.2f}%")
        
        return metrics
    
    def generate_random_predictions(self) -> pd.DataFrame:
        """Generate random F1 predictions for comparison"""
        if self.results_df is None:
            raise ValueError("Results must be loaded first to generate random predictions")
        
        logger.info("Generating random F1 predictions")
        random_df = self.results_df.copy()
        
        # Get unique drivers per race to assign positions
        races = random_df.groupby('race_id')
        new_data = []
        
        for race_id, race_data in races:
            drivers = race_data['driver'].tolist()
            positions = list(range(1, len(drivers) + 1))
            random.shuffle(positions)
            
            for i, (_, row) in enumerate(race_data.iterrows()):
                new_row = row.copy()
                new_row['predicted_position'] = positions[i]
                
                # Calculate points based on position
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                new_row['predicted_points'] = points_map.get(positions[i], 0)
                
                new_data.append(new_row)
        
        return pd.DataFrame(new_data)

class HTMLReportGenerator:
    """Generate HTML reports for prediction analysis"""
    
    @staticmethod
    def generate_report(analyzer: BaseSportsAnalyzer, sport_type: str, 
                       actual_metrics: Dict[str, PredictionMetrics],
                       random_metrics: Dict[str, PredictionMetrics],
                       output_path: str = "prediction_report.html"):
        """Generate comprehensive HTML report"""
        
        logger.info(f"Generating HTML report for {sport_type}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{sport_type.title()} Prediction Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-card h3 {{ margin: 0 0 15px 0; font-size: 1.2em; }}
                .metric-row {{ display: flex; justify-content: space-between; margin: 8px 0; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.2); }}
                .metric-label {{ font-weight: bold; }}
                .metric-value {{ font-family: 'Courier New', monospace; }}
                .comparison-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .comparison-table th, .comparison-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                .comparison-table th {{ background-color: #4CAF50; color: white; }}
                .comparison-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .better {{ color: #4CAF50; font-weight: bold; }}
                .worse {{ color: #f44336; font-weight: bold; }}
                .summary {{ background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .timestamp {{ color: #666; font-size: 0.9em; text-align: center; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{sport_type.title()} Prediction Analysis Report</h1>
                    <p>Comprehensive analysis of prediction accuracy vs actual results</p>
                </div>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>This report analyzes the accuracy of {sport_type} predictions compared to actual results, 
                    including comparison with random predictions to establish baseline performance.</p>
                    <p><strong>Total Predictions Analyzed:</strong> {actual_metrics[list(actual_metrics.keys())[0]].total_predictions}</p>
                </div>
        """
        
        # Add metrics cards
        html_content += '<div class="metrics-grid">'
        
        for metric_type, metrics in actual_metrics.items():
            html_content += f'''
                <div class="metric-card">
                    <h3>{metric_type.replace('_', ' ').title()} - Actual Predictions</h3>
                    <div class="metric-row">
                        <span class="metric-label">RMS Error:</span>
                        <span class="metric-value">{metrics.rms_error:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Mean Absolute Error:</span>
                        <span class="metric-value">{metrics.mean_absolute_error:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Exact Matches:</span>
                        <span class="metric-value">{metrics.exact_matches}/{metrics.total_predictions}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Accuracy:</span>
                        <span class="metric-value">{metrics.accuracy_percentage:.2f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Correlation:</span>
                        <span class="metric-value">{metrics.correlation_coefficient:.3f}</span>
                    </div>
                </div>
            '''
        
        html_content += '</div>'
        
        # Add comparison table
        html_content += '''
            <h2>Comparison with Random Predictions</h2>
            <table class="comparison-table">
                <tr>
                    <th>Metric Type</th>
                    <th>Prediction Accuracy</th>
                    <th>Random Accuracy</th>
                    <th>Improvement</th>
                </tr>
        '''
        
        for metric_type in actual_metrics.keys():
            actual_acc = actual_metrics[metric_type].accuracy_percentage
            random_acc = random_metrics[metric_type].accuracy_percentage
            improvement = actual_acc - random_acc
            improvement_class = "better" if improvement > 0 else "worse"
            
            html_content += f'''
                <tr>
                    <td>{metric_type.replace('_', ' ').title()}</td>
                    <td>{actual_acc:.2f}%</td>
                    <td>{random_acc:.2f}%</td>
                    <td class="{improvement_class}">{improvement:+.2f}%</td>
                </tr>
            '''
        
        html_content += f'''
            </table>
            
            <div class="timestamp">
                <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
        </body>
        </html>
        '''
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")

def analyze_predictions(sport_type: str, predictions_file: str, results_file: str, output_file: str = None):
    """Main function to analyze predictions"""
    
    if sport_type.lower() == 'football':
        analyzer = FootballAnalyzer()
    elif sport_type.lower() == 'f1':
        analyzer = F1Analyzer()
    else:
        raise ValueError("sport_type must be 'football' or 'f1'")
    
    try:
        # Load data
        analyzer.load_predictions(predictions_file)
        analyzer.load_results(results_file)
        
        # Calculate metrics for actual predictions
        actual_metrics = analyzer.calculate_metrics()
        
        # Generate and analyze random predictions
        random_predictions = analyzer.generate_random_predictions()
        original_predictions = analyzer.predictions_df
        analyzer.predictions_df = random_predictions
        analyzer.merged_data = None  # Reset merged data
        random_metrics = analyzer.calculate_metrics()
        
        # Restore original predictions
        analyzer.predictions_df = original_predictions
        analyzer.merged_data = None
        
        # Generate HTML report
        if output_file is None:
            output_file = f"{sport_type}_prediction_report.html"
        
        HTMLReportGenerator.generate_report(
            analyzer, sport_type, actual_metrics, random_metrics, output_file
        )
        
        logger.info("Analysis completed successfully")
        return actual_metrics, random_metrics
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Example for football predictions
    # analyze_predictions('football', 'football_predictions.csv', 'football_results.csv')
    
    # Example for F1 predictions  
    # analyze_predictions('f1', 'f1_predictions.csv', 'f1_results.csv')
    
    print("Sports Prediction Analysis System Ready!")
    print("Use analyze_predictions(sport_type, predictions_file, results_file) to run analysis")