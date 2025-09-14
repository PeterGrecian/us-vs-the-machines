import pandas as pd

# Path to the CSV file
file_path = 'data/f1-20250907.csv'

# Read the file as text
with open(file_path, 'r') as f:
    lines = f.readlines()

sections = {}
current_section = None
section_lines = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    # Detect section header
    if line and not line[0].isdigit() and ',' not in line:
        if current_section and section_lines:
            # Save previous section
            df = pd.read_csv(pd.compat.StringIO('\n'.join(section_lines)))
            sections[current_section] = df
        current_section = line
        section_lines = []
    elif current_section:
        section_lines.append(line)

# Save last section
if current_section and section_lines:
    df = pd.read_csv(pd.compat.StringIO('\n'.join(section_lines)))
    sections[current_section] = df

# Example: print all DataFrames
for name, df in sections.items():
    print(f'\nSection: {name}')
    print(df)
