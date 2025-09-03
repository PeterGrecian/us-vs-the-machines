
from scipy.stats import spearmanr

# Assume these are your dictionaries of drivers and their positions
predicted_positions = {
    "Lewis Hamilton": 1,
    "Max Verstappen": 2,
    "Charles Leclerc": 3,
    "Lando Norris": 4,
    "George Russell": 5
}

actual_positions = {
    "Max Verstappen": 1,
    "Lando Norris": 2,
    "Lewis Hamilton": 3,
    "Charles Leclerc": 4,
    "George Russell": 5
}

# Get a consistent order of drivers to compare
drivers = actual_positions.keys()

# Create ordered lists of positions
predicted_ranks = [predicted_positions[driver] for driver in drivers]
actual_ranks = [actual_positions[driver] for driver in drivers]

# The lists are now ordered based on the 'drivers' list
# predicted_ranks will be [1, 2, 3, 4, 5]
# actual_ranks will be [3, 1, 4, 2, 5]
from scipy.stats import spearmanr

# Assume these are your dictionaries of drivers and their positions
predicted_positions = {
    "Lewis Hamilton": 1,
    "Max Verstappen": 2,
    "Charles Leclerc": 3,
    "Lando Norris": 4,
    "George Russell": 5
}

actual_positions = {
    "Max Verstappen": 1,
    "Lando Norris": 2,
    "Lewis Hamilton": 3,
    "Charles Leclerc": 4,
    "George Russell": 5
}

# Get a consistent order of drivers to compare
drivers = actual_positions.keys()

# Create ordered lists of positions
predicted_ranks = [predicted_positions[driver] for driver in drivers]
actual_ranks = [actual_positions[driver] for driver in drivers]

# The lists are now ordered based on the 'drivers' list
# predicted_ranks will be [1, 2, 3, 4, 5]
# actual_ranks will be [3, 1, 4, 2, 5]
correlation, p_value = spearmanr(predicted_ranks, actual_ranks)
print(f"Spearman's Rank Correlation: {correlation}, p-value: {p_value}")

