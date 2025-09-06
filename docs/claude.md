I'll create a comprehensive Python system for analyzing football match and F1 race predictions. This will include classes for reading CSV files, calculating various metrics, and generating HTML reports.I've created a comprehensive sports prediction analysis system with the following key components:

## Main Classes:

1. **BaseSportsAnalyzer** - Abstract base class defining the interface
2. **FootballAnalyzer** - Handles football match predictions (scores, results)
3. **F1Analyzer** - Handles F1 race predictions (positions, points)
4. **HTMLReportGenerator** - Creates detailed HTML reports

## Key Features:

### Data Handling:
- **CSV Loading**: Robust CSV reading with error handling and column validation
- **Data Merging**: Intelligent merging of predictions and results
- **Missing Data**: Automatic calculation of derived fields (like win/draw/loss for football)

### Metrics Calculated:
- **RMS Error**: Root Mean Square error for numerical predictions
- **Mean Absolute Error**: Average absolute difference
- **Exact Matches**: Perfect predictions count
- **Accuracy Percentage**: Success rate
- **Correlation Coefficient**: Linear relationship strength

### Comparison Features:
- **Random Baseline**: Generates random predictions for comparison
- **Performance Analysis**: Shows how much better (or worse) predictions are vs random

### Expected CSV Formats:

**Football Predictions:**
```csv
match_id,team_a,team_b,predicted_score_a,predicted_score_b,predicted_result
1,Arsenal,Chelsea,2,1,win
```

**Football Results:**
```csv
match_id,actual_score_a,actual_score_b,actual_result
1,1,1,draw
```

**F1 Predictions:**
```csv
race_id,driver,predicted_position,predicted_points
1,Hamilton,1,25
```

**F1 Results:**
```csv
race_id,driver,actual_position,actual_points
1,Hamilton,3,15
```

## Usage:

```python
# Analyze football predictions
actual_metrics, random_metrics = analyze_predictions(
    'football', 
    'football_predictions.csv', 
    'football_results.csv'
)

# Analyze F1 predictions
actual_metrics, random_metrics = analyze_predictions(
    'f1', 
    'f1_predictions.csv', 
    'f1_results.csv'
)
```

The system will generate a beautiful HTML report showing all metrics, comparisons with random predictions, and detailed analysis. The logger provides detailed debugging information throughout the process.