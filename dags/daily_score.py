"""daily_score: score each cutoff, compute its PSI against the frozen reference, alert on drift.

Decisions in docs/pipeline.md (P5). One run per cutoff, 2019-10-15 to 2019-12-01; the cutoff is
data_interval_end, the morning the run is for in both scheduling modes.
"""

from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# The repo root is the parent of dags/, so the DAG runs from any checkout (WSL or n100b).
REPO = str(Path(__file__).resolve().parents[1])
PYTHON = "PTE_ENV=wsl ~/.venvs/pte/bin/python"
CUTOFF = "{{ data_interval_end | ds }}"

with DAG(
    dag_id="daily_score",
    schedule="@daily",
    start_date=pendulum.datetime(2019, 10, 15, tz="UTC"),
    end_date=pendulum.datetime(2019, 12, 1, tz="UTC"),
    catchup=True,
    max_active_runs=1,
):
    score = BashOperator(
        task_id="score",
        bash_command=f"{PYTHON} scripts/score_users.py --cutoff {CUTOFF}",
        cwd=REPO,
    )

    psi = BashOperator(
        task_id="psi",
        bash_command=f"{PYTHON} scripts/daily_psi.py --cutoff {CUTOFF}",
        cwd=REPO,
    )

    alert = BashOperator(
        task_id="alert",
        bash_command=f"{PYTHON} scripts/psi_alert.py --cutoff {CUTOFF}",
        cwd=REPO,
    )

    score >> psi >> alert
