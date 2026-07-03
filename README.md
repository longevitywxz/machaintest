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

If those files are missing, `prepare_data.py` uses `individual+household+electric+power+consumption.zip` from the repository root when present. Otherwise it downloads the UCI Individual Household Electric Power Consumption dataset and creates a reproducible daily split.

Monthly weather variables are downloaded from data.gouv / Meteo-France for department 92 (`MENSQ_92_previous-1950-2024.csv.gz`) and merged by year-month. The merged columns are `RR`, `NBJRR1`, `NBJRR5`, `NBJRR10`, and `NBJBROU`.

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
- Daily aggregation follows the assignment: power and sub-metering variables are summed; voltage and intensity are averaged; monthly weather columns are merged by month.
- Missing minute-level values are handled by numeric coercion and daily interpolation.
