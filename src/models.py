class RaceStandings:
    def __init__(self, description, drivers_and_positions):
        self.description = description
        self.drivers_and_positions = drivers_and_positions

class PredictedRaceStandings(RaceStandings):
    def __init__(self, description, drivers_and_positions, prediction_source):
        super().__init__(description, drivers_and_positions)
        self.prediction_source = prediction_source

class ActualRaceResults(RaceStandings):
    def __init__(self, description, drivers_and_positions, race_time):
        super().__init__(description, drivers_and_positions)
        self.race_time = race_time
        
class PredictionAnalysis:
    def __init__(self, predicted_standings, actual_results):
        self.predicted = predicted_standings
        self.actual = actual_results
        self.mae_score = self.calculate_mae()
        self.spearman_score = self.calculate_spearman()

    def calculate_mae(self):
        # Calculation logic for MAE
        pass

    def calculate_spearman(self):
        # Calculation logic for Spearman's Rank Correlation
        pass
