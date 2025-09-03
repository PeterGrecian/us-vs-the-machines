# us-vs-themachines

This project analyzes the accuracy of race predictions by comparing them to the actual race results using statistical metrics.


### Prerequisites

  * Python 3.6+
  * `pip`

### Installation

To set up the project, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    ```

2.  **Navigate to the project directory**:

    ```bash
    cd race_analysis
    ```

3.  **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the analysis and see the results, execute the main script from the root directory of the project.

```bash
python src/main.py
```

### File Structure

This project follows a standard layout for Python applications.

```
race_analysis/
├── data/                  # Input files (race results, predictions)
├── src/                   # Source code
│   ├── models.py          # Data models (RaceStandings classes)
│   ├── analysis.py        # Analysis logic (PredictionAnalysis class)
│   └── main.py            # Main script to run the analysis
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

### Author

* **Project:** [Us vs. The Machines](https://github.com/PeterGrecian/us-vs-the-machines)
* **Author:** [Peter Grecian](https://github.com/PeterGrecian)
* **Personal Website:** [petergrecian.co.uk](https://w3.petergrecian.co.uk)

