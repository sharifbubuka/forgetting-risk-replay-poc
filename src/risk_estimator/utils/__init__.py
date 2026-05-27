_EXPORT_MODULES = {
    "set_seed": ".seed",
    "get_project_root": ".paths",
    "get_results_dir": ".paths",
    "get_figures_dir": ".paths",
    "get_tables_dir": ".paths",
    "ensure_results_dirs": ".paths",
    "save_table": ".tables",
    "save_current_figure": ".plotting",
    "plot_memory_distribution": ".plotting",
    "plot_bar_metric": ".plotting",
    "plot_accuracy_matrix_over_time": ".plotting",
    "plot_temporal_summary": ".plotting",
    "plot_risk_vs_forgetting": ".plotting",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name in _EXPORT_MODULES:
        from importlib import import_module

        module = import_module(_EXPORT_MODULES[name], __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
