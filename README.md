# Attraction and Repulsion

Run from this directory with Python 3.13.9 (CPU).

```bash
python -m pip install -r requirements.txt
python prepare_data.py --input-dir /path/to/nsfc_literature
python build_panel.py
python relation_model.py
python emergence_model.py
```

Input: `nsfc_final_projects_F_YYYY.jsonl` files covering 1987–2024, with fields `project_id`, `conclusion_year`, `application_code` (F-code), and original `keywords` (a list or semicolon-separated string).

Outputs are written to `runs/`: `relations/` contains relation backtests and forecasts; `emergence/` contains emerging-direction backtests and forecasts. The fixed horizons are 1, 5 and 10 years from each origin, with final forecasts originating in 2024.

Resume interrupted relation fitting:

```bash
python relation_model.py --resume
```

For another emergence run, use a fresh output directory:

```bash
python emergence_model.py --output runs/emergence_rerun
```
