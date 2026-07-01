# Household Power Forecasting

Machine learning course project for multivariate household electric power consumption forecasting.

The project trains and compares:

- LSTM
- Transformer encoder
- CNN + Transformer encoder improvement

Each model is trained separately for 90-day and 365-day forecasting from the previous 90 days. Metrics are MSE and MAE over five random seeds.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

If the course-provided files are available, place them here:

```text
data/raw/train.csv
data/raw/test.csv
```

The scripts also accept the typo from the assignment:

```text
data/raw/tes.csv
```

If those files are missing, `prepare_data.py` downloads the UCI Individual Household Electric Power Consumption dataset and creates a reproducible daily split.

## Run

```powershell
python src/prepare_data.py
python src/train.py --horizons 90 365 --seeds 0 1 2 3 4 --epochs 30
python src/make_report.py
```

Outputs:

- `outputs/metrics.csv`
- `outputs/metrics_summary.csv`
- `outputs/figures/*.png`
- `report/report.md`

## Notes

- The long-horizon and short-horizon models are trained independently.
- Daily aggregation follows the assignment: power and sub-metering variables are summed; voltage and intensity are averaged; weather-like columns are carried by first valid daily value when present.
- Missing minute-level values are handled by numeric coercion and daily interpolation.
