from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_results_dir() -> Path:
    return get_project_root() / "results"


def get_figures_dir() -> Path:
    return get_results_dir() / "figures"


def get_tables_dir() -> Path:
    return get_results_dir() / "tables"


def ensure_results_dirs() -> None:
    """Create results folders if missing."""
    get_figures_dir().mkdir(parents=True, exist_ok=True)
    get_tables_dir().mkdir(parents=True, exist_ok=True)