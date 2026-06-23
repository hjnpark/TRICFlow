# TRICFlow tests

Unit tests use fixtures under `tests/data/` and `tests/data/pathway_assembly/`.
No live QM calculations are run in the default suite.

## Layout

| File | What it checks |
|------|----------------|
| `test_workflow_cache.py` | QM chemistry fingerprinting, cache reuse/fail-fast |
| `test_pathway_assembly.py` | Five-segment pathway assembly from `tests/data/pathway_assembly/refined_steps/` |
| `test_workflow_calc_kwargs.py` | `TRICWorkflow` kwargs and geomeTRIC command building |
| `test_optimize_ts_failure.py` | TS optimization failure handling |
| `test_neb_barrierless.py` | Barrierless NEB segments (no TS climb) |
| `test_postopt_trajectory.py` | Post-opt IRC trajectory concatenation |

## Running

```bash
pytest tests/ -q

# With coverage
pytest tests/ --cov=tricflow --cov-report=term-missing
```