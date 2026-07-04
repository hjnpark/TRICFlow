"""Local sequential reaction-pathway workflow driven by geomeTRIC (TRIC coordinates)."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from contextlib import contextmanager
from typing import List, Optional, Union, Dict, Any, Sequence, Tuple, Literal, FrozenSet
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING

from tricflow.errors import OptimizationError, WorkflowError

_VERBOSITY = 0
_STEP_LABEL: Optional[str] = None


@contextmanager
def _log_context(*, verbosity: Optional[int] = None, step: Optional[str] = None):
    """Temporarily override module logging verbosity and/or step label."""
    global _VERBOSITY, _STEP_LABEL
    prev = (_VERBOSITY, _STEP_LABEL)
    if verbosity is not None:
        _VERBOSITY = int(verbosity)
    if step is not None:
        _STEP_LABEL = step
    try:
        yield
    finally:
        _VERBOSITY, _STEP_LABEL = prev


def _log_prefix() -> str:
    if _STEP_LABEL:
        return f"[TRICFlow:{_STEP_LABEL}]"
    return "[TRICFlow]"


def _log(msg: str, *, level: int = 0) -> None:
    """Emit a message when ``_VERBOSITY >= level`` (0 = summary, 1 = detail)."""
    if _VERBOSITY >= level:
        print(f"{_log_prefix()} {msg}")


def _log_warn(msg: str) -> None:
    print(f"{_log_prefix()} WARNING: {msg}")


def _pop_verbose(kwargs: Dict[str, Any]) -> int:
    """Remove and return an explicit ``verbose`` override from *kwargs*."""
    value = kwargs.pop("verbose", _VERBOSITY)
    return int(value)


def _append_verbose_to_cmd(cmd: list, verbose: int) -> None:
    """Forward verbosity to geomeTRIC when ``verbose >= 1``."""
    if verbose >= 1:
        cmd += ["--verbose", str(verbose)]

if TYPE_CHECKING:
    from geometric.molecule import Molecule as GeoMolecule

# Lazy import so the module can be used for prototyping even if geometric
# is not installed yet.
_GeoMolecule = None

def _get_geo_molecule():
    global _GeoMolecule
    if _GeoMolecule is None:
        try:
            from geometric.molecule import Molecule as _M
            _GeoMolecule = _M
        except ImportError:
            _GeoMolecule = None
    return _GeoMolecule


def _get_frame_list(mol):
    """
    Split a geomeTRIC Molecule into single-frame Molecule objects.

    A single-frame input is returned as a one-element list.
    """
    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for _get_frame_list.")

    if not hasattr(mol, "xyzs"):
        raise TypeError(f"Expected a geomeTRIC Molecule, got {type(mol)!r}")

    n_frames = len(mol.xyzs)
    if n_frames > 1:
        frames = []
        for i in range(n_frames):
            frame_mol = GeoM()
            frame_mol.elem = list(mol.elem)
            frame_mol.xyzs = [mol.xyzs[i]]
            if hasattr(mol, "charge"):
                frame_mol.charge = mol.charge
            if hasattr(mol, "multiplicity"):
                frame_mol.multiplicity = mol.multiplicity
            frames.append(frame_mol)
        return frames
    return [mol]


def _parse_cmd_dict(s: str) -> dict:
    """Parse a geometric command line into a param dict, ignoring invocation prefix."""
    if not s:
        return {}
    parts = shlex.split(s)
    # Skip leading python, -m, script path, module name etc.
    i = 0
    while i < len(parts):
        p = parts[i]
        if (p.endswith(".py") or
            "python" in p.lower() or
            "/geometric/" in p or
            p == "-m" or
            p == "geometric.optimize" or "geometric-neb" in p or
            "geometric.optimize" in p):
            i += 1
            continue
        break
    d = {}
    # Next non-flag token is usually the input file
    if i < len(parts) and not parts[i].startswith("--"):
        d["input"] = parts[i]
        i += 1
    while i < len(parts):
        if parts[i].startswith("--"):
            key = parts[i][2:]
            i += 1
            if i < len(parts) and not parts[i].startswith("--"):
                d[key] = parts[i]
                i += 1
            else:
                d[key] = True
        else:
            i += 1
    return d


def _cmds_equivalent(d1: dict, d2: dict) -> bool:
    """Compare command dicts. --nt is allowed to differ (parallelism only)."""
    keys = set(d1) | set(d2)
    differing = []
    for k in keys:
        if k == "nt":
            continue  # --nt does not affect optimization result
        if d1.get(k) != d2.get(k):
            differing.append(k)
    return len(differing) == 0


_CHEM_KEYS = ("method", "basis", "charge", "mult")


def _normalize_chem_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _extract_psi4_charge_mult_from_template(psi4_temp) -> Tuple[str, str]:
    for i, line in enumerate(psi4_temp):
        if "molecule" in line.lower():
            if i + 1 < len(psi4_temp):
                parts = psi4_temp[i + 1].split()
                if (
                    len(parts) == 2
                    and parts[0].lstrip("+-").isdigit()
                    and parts[1].isdigit()
                ):
                    return parts[0], parts[1]
            break
    return "", ""


def _extract_qm_chemistry_text(input_text: str) -> Dict[str, str]:
    """Fallback parser for Psi4-style QM templates when geomeTRIC is unavailable."""
    method = ""
    basis = ""
    charge = ""
    mult = ""
    in_molecule = False
    for raw_line in input_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"molecule\s*\{", line, flags=re.IGNORECASE):
            in_molecule = True
            continue
        if in_molecule:
            if "}" in line:
                in_molecule = False
                continue
            parts = line.split()
            if (
                len(parts) == 2
                and parts[0].lstrip("+-").isdigit()
                and parts[1].isdigit()
            ):
                charge, mult = parts[0], parts[1]
                continue
        basis_m = re.match(r"set\s+basis\s+(.+)", line, flags=re.IGNORECASE)
        if basis_m:
            basis = basis_m.group(1).strip().lower()
            continue
        charge_m = re.match(r"set\s+charge\s+(.+)", line, flags=re.IGNORECASE)
        if charge_m:
            charge = charge_m.group(1).strip()
            continue
        mult_m = re.match(r"set\s+multiplicity\s+(.+)", line, flags=re.IGNORECASE)
        if mult_m:
            mult = mult_m.group(1).strip()
            continue
        for pattern in (
            r"(?:gradient|energy|hessian)\s*\(\s*['\"]([^'\"]+)['\"]",
            r"set\s+dft_functional\s+(.+)",
        ):
            method_m = re.search(pattern, line, flags=re.IGNORECASE)
            if method_m:
                method = method_m.group(1).strip().lower()
                break
    return {
        "method": method,
        "basis": basis,
        "charge": _normalize_chem_value(charge),
        "mult": _normalize_chem_value(mult),
    }


def _extract_qm_chemistry(
    input_path: Union[str, Path],
    qm_program: str = "psi4",
) -> Dict[str, str]:
    """Return normalized method/basis/charge/mult from a QM input file."""
    path = Path(input_path)
    if not path.is_file():
        return {key: "" for key in _CHEM_KEYS}

    try:
        from geometric.prepare import get_molecule_engine

        _, engine = get_molecule_engine(engine=qm_program, input=str(path))
        molecule = engine.M
        chem = {
            "method": _normalize_chem_value(getattr(engine, "method", None)),
            "basis": _normalize_chem_value(getattr(engine, "basis", None)),
            "charge": _normalize_chem_value(getattr(molecule, "charge", None)),
            "mult": _normalize_chem_value(getattr(molecule, "mult", None)),
        }
        if chem["charge"] == "":
            chem["charge"] = _normalize_chem_value(molecule.Data.get("charge"))
        if chem["mult"] == "":
            chem["mult"] = _normalize_chem_value(molecule.Data.get("mult"))

        if qm_program.lower() == "psi4" and (not chem["charge"] or not chem["mult"]):
            chg, mult = _extract_psi4_charge_mult_from_template(
                getattr(engine, "psi4_temp", []),
            )
            if chg and not chem["charge"]:
                chem["charge"] = _normalize_chem_value(chg)
            if mult and not chem["mult"]:
                chem["mult"] = _normalize_chem_value(mult)

        if any(chem.values()):
            return chem
    except Exception:
        pass

    return _extract_qm_chemistry_text(path.read_text())


def _qm_chemistry_matches(
    cached_input: Union[str, Path],
    current_input: Union[str, Path],
    *,
    qm_program: str = "psi4",
) -> bool:
    """True when method/basis/charge/mult are unchanged; falls back to full-file equality."""
    cached_path = Path(cached_input)
    current_path = Path(current_input)
    cached = _extract_qm_chemistry(cached_path, qm_program)
    current = _extract_qm_chemistry(current_path, qm_program)
    if cached["method"] and cached["basis"] and current["method"] and current["basis"]:
        for key in _CHEM_KEYS:
            cached_val = cached.get(key, "")
            current_val = current.get(key, "")
            if key in ("charge", "mult"):
                if cached_val or current_val:
                    if cached_val != current_val:
                        return False
            elif cached_val != current_val:
                return False
        return True
    return cached_path.read_text() == current_path.read_text()


def _log_cmd_diff_but_reusing(label: str, prev_cmd: str, curr_cmd: str) -> None:
    _log(
        f"Command line differs for {label} but prior run converged with unchanged "
        "method/basis/charge/mult; reusing cache.",
        level=1,
    )
    _log(f"  Cached: {prev_cmd}", level=1)
    _log(f"  Current: {curr_cmd}", level=1)


def _rename_conflicting_cache_dir(path: Path) -> None:
    """Move aside an existing run directory so a fresh calculation can proceed."""
    if not path.exists():
        return
    k = 0
    while True:
        candidate = path.parent / f"{path.name}_{k}"
        if not candidate.exists():
            break
        k += 1
    _log(f"Renaming old cache directory {path} -> {candidate}", level=1)
    try:
        path.rename(candidate)
    except Exception as err:
        _log_warn(f"Failed to rename old cache directory: {err}")


def _failed_cache_message(label: str, log_path: Optional[Path]) -> str:
    log_ref = log_path.name if log_path is not None else "log file"
    return (
        f"{label}: prior calculation did not converge and the requested "
        f"command is unchanged. Check {log_ref}."
    )


def _raise_failed_cache(
    label: str,
    log_path: Optional[Path],
    *,
    error_cls: type = OptimizationError,
) -> None:
    raise error_cls(_failed_cache_message(label, log_path))


def _same_command_as_prior(
    prev_cmd_str: Optional[str],
    intended_cmd_str: str,
) -> bool:
    if not prev_cmd_str:
        return False
    return _cmds_equivalent(
        _parse_cmd_dict(prev_cmd_str),
        _parse_cmd_dict(intended_cmd_str),
    )


def _converged_cache_decision(
    *,
    log_path: Optional[Path],
    cached_input: Path,
    current_input: Path,
    prev_cmd_str: Optional[str],
    intended_cmd_str: str,
    label: str,
    qm_program: str = "psi4",
    allow_artifact_fallback: bool = False,
    artifact_ok: bool = False,
) -> str:
    """
    Decide whether a prior geomeTRIC run may be reused.

    Returns ``\"reuse\"``, ``\"rerun\"``, ``\"fail\"``, or ``\"invalidate\"``
    (QM chemistry changed). A converged log is reused even when argv differs,
    unless QM chemistry changed. When a prior log exists without a convergence
    marker and the requested command is unchanged, returns ``\"fail\"`` so the
    caller can raise instead of repeating a known-bad calculation.
    """
    if not cached_input.is_file() or not current_input.is_file():
        return "rerun"
    if not _qm_chemistry_matches(cached_input, current_input, qm_program=qm_program):
        _log(
            f"QM method/basis/charge/mult changed for {label}; cache invalid.",
            level=1,
        )
        return "invalidate"

    if log_path is not None and log_path.is_file():
        if _check_log_for_caching(log_path):
            if prev_cmd_str:
                prev_dict = _parse_cmd_dict(prev_cmd_str)
                curr_dict = _parse_cmd_dict(intended_cmd_str)
                if not _cmds_equivalent(prev_dict, curr_dict):
                    _log_cmd_diff_but_reusing(label, prev_cmd_str, intended_cmd_str)
            return "reuse"
        if _same_command_as_prior(prev_cmd_str, intended_cmd_str):
            return "fail"
    if allow_artifact_fallback and artifact_ok:
        return "reuse"
    return "rerun"


def _parse_energy_hartree(
    opt_mol=None,
    comment: Optional[str] = None,
    log_path: Optional[Union[str, Path]] = None,
    combined_log: Optional[str] = None,
) -> Optional[float]:
    """
    Extract the final optimization energy (Hartree) from geomeTRIC outputs.

    geomeTRIC may record energy as qm_energies, comment lines like
    ``Energy = -1.23``, or ``Iteration 7 Energy -1.23``, or in optimize.log /
    stdout as ``E (change) = -1.23`` on the last step line.
    """
    if opt_mol is not None and getattr(opt_mol, "qm_energies", None):
        try:
            return float(opt_mol.qm_energies[-1])
        except Exception:
            pass

    if comment is None and opt_mol is not None and getattr(opt_mol, "comms", None):
        comment = opt_mol.comms[-1] if opt_mol.comms else None

    if isinstance(comment, str):
        for pattern in (
            r"Energy\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            r"Energy\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            r"Iteration\s+\d+\s+Energy\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        ):
            m = re.search(pattern, comment, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass

    def _scan_text(text: str) -> Optional[float]:
        for line in reversed(text.splitlines()):
            m = re.search(
                r"E\s*\(change\)\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
                line,
            )
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            m = re.search(r"Energy\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    if log_path is not None:
        lp = Path(log_path)
        if lp.is_file():
            energy = _scan_text(lp.read_text())
            if energy is not None:
                return energy

    if combined_log:
        return _scan_text(combined_log)
    return None


_LOG_CACHE_CONVERGENCE_MARKERS = (
    "Converged!",
    "IRC backward direction reached maximum iteration number",
    "Optimization Converged",
    "Maximum optimization cycles reached.", # For NEB
)


def _check_log_for_caching(
    log_path: Path,
    markers: tuple = _LOG_CACHE_CONVERGENCE_MARKERS,
) -> bool:
    """True if any *markers* appear when scanning *log_path* from the end.

    Used to decide whether a prior geomeTRIC run finished successfully enough
    to reuse cached artifacts.  Scans from EOF backward so post-convergence
    output (Hessian, vibrational analysis, IRC backward leg, etc.) does not
    hide an earlier convergence line.  A fixed tail window (e.g. last 20 lines)
    is unreliable for TS and IRC logs in particular.
    """
    if not log_path.is_file():
        return False
    with open(log_path, "r") as lf:
        for line in reversed(lf.readlines()):
            for marker in markers:
                if marker in line:
                    return True
    return False


def _find_converged_log_in_dir(
    run_dir: Path,
    *,
    preferred: Optional[Path] = None,
    markers: tuple = _LOG_CACHE_CONVERGENCE_MARKERS,
) -> Optional[Path]:
    """Return a log under *run_dir* that contains a convergence *markers* entry."""
    candidates: List[Path] = []
    if preferred is not None:
        candidates.append(preferred)
    if run_dir.is_dir():
        candidates.extend(sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True))
    seen = set()
    for log_path in candidates:
        resolved = log_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _check_log_for_caching(log_path, markers=markers):
            return log_path
    return None


def _optimization_artifact_cache_hit(
    frame_dir: Path,
    user_input: Path,
    initial_xyz,
    prefix: Optional[str],
    GeoM,
    *,
    qm_program: str = "psi4",
) -> bool:
    """True when a prior optimization output exists for the current input geometry."""
    dest_input = frame_dir / user_input.name
    coords_xyz = frame_dir / "coords.xyz"
    opt_path = frame_dir / _prefix_optim_xyz(prefix)
    if not (dest_input.is_file() and coords_xyz.is_file() and opt_path.is_file()):
        return False
    if not _qm_chemistry_matches(dest_input, user_input, qm_program=qm_program):
        return False
    try:
        cached_coords = GeoM(str(coords_xyz))
        if not cached_coords.xyzs or not np.allclose(
            cached_coords.xyzs[0], np.asarray(initial_xyz, dtype=float), atol=1e-8,
        ):
            return False
        opt_mol = GeoM(str(opt_path))
        if not getattr(opt_mol, "xyzs", None):
            return False
        final_xyz = opt_mol.xyzs[-1] if len(opt_mol.xyzs) > 1 else opt_mol.xyzs[0]
        return not np.allclose(final_xyz, cached_coords.xyzs[0], atol=1e-8)
    except Exception:
        return False


_GEOMETRIC_CMD_LOG_MARKERS = (
    "geometric-optimize called with the following command line:",
    "geometric-neb called with the following command line:",
    "geometric-interpolate called with the following command line:",
)


def _read_logged_geometric_cmd(
    log_path: Optional[Path],
    *,
    run_dir: Optional[Path] = None,
) -> Optional[str]:
    """Return the command line recorded in a geomeTRIC log or ``last_command.txt``."""
    if log_path is not None and log_path.is_file():
        with open(log_path, "r") as lf:
            lines = lf.readlines()
        for i, line in enumerate(lines):
            if any(marker in line for marker in _GEOMETRIC_CMD_LOG_MARKERS):
                if i + 1 < len(lines):
                    cmd = lines[i + 1].strip()
                    if cmd:
                        return cmd
    if run_dir is not None:
        last_cmd = run_dir / "last_command.txt"
        if last_cmd.is_file():
            return last_cmd.read_text().strip() or None
    return None


def _write_last_command(run_dir: Path, cmd: Sequence[str]) -> None:
    (run_dir / "last_command.txt").write_text(" ".join(cmd))


def _prefix_optim_xyz(prefix: str) -> str:
    return f"{prefix}_optim.xyz"


def _prefix_log(prefix: str) -> str:
    return f"{prefix}.log"


def _prefix_neb_ts_climb(prefix: str) -> str:
    return f"{prefix}.tsClimb.xyz"


def _prefix_irc_traj(prefix: str) -> str:
    return f"{prefix}_irc.xyz"


def _prefix_irc_postopt_traj(prefix: str) -> str:
    return f"{prefix}_irc_postopt.xyz"


def _write_postopt_irc_trajectory(traj, base: Path, artifact_prefix: str) -> Path:
    """Write the opt + IRC + opt trajectory beside the raw IRC artifact."""
    out_path = base / _prefix_irc_postopt_traj(artifact_prefix)
    traj.write(str(out_path))
    return out_path


def _resolve_run_dir(run_dir: Optional[Union[str, Path]]) -> Path:
    """Return *run_dir* when given; otherwise the process cwd."""
    if run_dir is not None:
        return Path(run_dir)
    return Path.cwd()


def _geometric_argv_path(path: Union[str, Path], cwd: Union[str, Path]) -> str:
    """Return a path string for geomeTRIC argv when subprocess ``cwd`` is set."""
    resolved = Path(path).resolve()
    run_dir = Path(cwd).resolve()
    try:
        return str(resolved.relative_to(run_dir))
    except ValueError:
        return str(resolved)


def _output_prefix(input_path: Union[str, Path], user_prefix: Optional[str] = None) -> str:
    """Artifact basename prefix: explicit *user_prefix* or geomeTRIC's input-file stem."""
    if user_prefix is not None:
        return user_prefix
    return Path(input_path).stem


ConvergeSpec = Union[str, Sequence[str]]


def _converge_tokens(spec: ConvergeSpec) -> List[str]:
    """Normalize a geomeTRIC ``--converge`` value to argv tokens."""
    if isinstance(spec, str):
        return shlex.split(spec)
    return list(spec)


def _append_converge_to_cmd(cmd: list, converge: Optional[ConvergeSpec]) -> None:
    """Append ``--converge`` and its tokens to a geomeTRIC optimize argv list."""
    if converge is None:
        return
    tokens = _converge_tokens(converge)
    if tokens:
        cmd += ["--converge", *tokens]


def _subprocess_timeout_kwargs(run_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{"timeout": ...}`` only when *run_kwargs* sets ``timeout`` explicitly."""
    timeout = run_kwargs.get("timeout")
    if timeout is not None:
        return {"timeout": timeout}
    return {}


GeometricJob = Literal["opt", "ts", "irc", "neb"]

# Keys consumed by TRICFlow / drivers — never forwarded as geomeTRIC CLI flags.
_TRICFLOW_GEOMETRIC_BLOCKLIST: FrozenSet[str] = frozenset({
    "timeout",
    "postopt_converge",
    "postopt_rmsd_threshold",
    "prefix",
    "qm_program",
    "input_file",
    "log_prefix",
    "direction",
    "hessian",
    "postopt",
    "recalc_hessian_on_failure",
    "n_images",
    "spring_constant",
    "nebk",
    "run_dir",
    "block_until_ms",
    "fast",
    "align_system",
    "align_frags",
})

# Set explicitly per job type before generic kwargs pass-through.
_JOB_EXPLICIT_GEOMETRIC_KEYS: Dict[str, FrozenSet[str]] = {
    "opt": frozenset({"coordsys", "nt", "nthreads", "maxiter", "converge", "verbose", "coords"}),
    "ts": frozenset({
        "coordsys", "nt", "nthreads", "maxiter", "converge", "verbose", "coords",
        "transition", "hessian",
    }),
    "irc": frozenset({
        "coordsys", "nt", "nthreads", "maxiter", "converge", "verbose", "coords",
        "trust", "irc", "irc_direction", "hessian",
    }),
    "neb": frozenset({
        "coordsys", "nt", "nthreads", "maxiter", "converge", "verbose",
        "images", "nebk", "neb", "method", "basis", "climb", "maxg", "avgg",
    }),
}


def _geometric_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _append_geometric_kwargs(
    cmd: list,
    kwargs: Dict[str, Any],
    *,
    job: GeometricJob,
    neb_use_maxcyc: bool = False,
) -> None:
    """Append ``--key value`` pairs for remaining geomeTRIC options in *kwargs*."""
    skip = _TRICFLOW_GEOMETRIC_BLOCKLIST | _JOB_EXPLICIT_GEOMETRIC_KEYS[job]
    for key, value in kwargs.items():
        if key in skip or value is None:
            continue
        if key == "maxiter" and job == "neb" and neb_use_maxcyc:
            cmd += ["--neb_maxcyc", _geometric_cli_value(value)]
            continue
        cmd += [f"--{key}", _geometric_cli_value(value)]


def _build_geometric_cmd(
    job: GeometricJob,
    input_path: Union[str, Path],
    coords_path: Union[str, Path],
    qm_program: str,
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    nt: Optional[int] = None,
    prefix: Optional[str] = None,
    hessian_path: Optional[Union[str, Path]] = None,
    irc_direction: str = "both",
    n_images: Optional[int] = None,
    spring_constant: Optional[float] = None,
) -> List[str]:
    """Build argv for geomeTRIC optimize (opt/ts/irc) or NEB (neb)."""
    import shutil as _shutil

    kw = dict(kwargs or {})
    inp = str(input_path)
    coords = str(coords_path)

    if job == "neb":
        neb_exe = _shutil.which("geometric-neb")
        coordsys = kw.get("coordsys", "tric")
        if neb_exe:
            cmd: List[str] = [neb_exe, inp, coords]
        else:
            cmd = [
                sys.executable, "-m", "geometric.optimize", inp,
                "--neb", "--coords", coords, "--coordsys", coordsys,
            ]
        cmd += ["--engine", qm_program]
        if qm_program != "psi4":
            if kw.get("method"):
                cmd += ["--method", str(kw["method"])]
            if kw.get("basis"):
                cmd += ["--basis", str(kw["basis"])]
        if n_images is not None:
            cmd += ["--images", str(n_images)]
        if spring_constant is not None:
            cmd += ["--nebk", str(spring_constant)]
        if prefix is not None:
            cmd += ["--prefix", prefix]
        if "climb" in kw:
            cmd += ["--climb", _geometric_cli_value(kw["climb"])]
        thread_count = nt if nt is not None else kw.get("nt") or kw.get("nthreads")
        if thread_count is not None:
            cmd += ["--nt", str(thread_count)]
        if "maxiter" in kw:
            if neb_exe:
                cmd += ["--neb_maxcyc", str(kw["maxiter"])]
            else:
                cmd += ["--maxiter", str(kw["maxiter"])]
        for flag in ("maxg", "avgg"):
            if flag in kw:
                cmd += [f"--{flag}", str(kw[flag])]
        _append_verbose_to_cmd(cmd, kw.get("verbose", _VERBOSITY))
        _append_geometric_kwargs(cmd, kw, job="neb", neb_use_maxcyc=bool(neb_exe))
        return cmd

    cmd = [sys.executable, "-m", "geometric.optimize", inp]
    cmd += ["--engine", qm_program]
    cmd += ["--coordsys", kw.get("coordsys", "tric")]
    cmd += ["--coords", coords]

    thread_count = nt if nt is not None else kw.get("nt") or kw.get("nthreads")
    if thread_count is not None:
        cmd += ["--nt", str(thread_count)]
    if "maxiter" in kw:
        cmd += ["--maxiter", str(kw["maxiter"])]
    if job == "irc" and "trust" in kw:
        cmd += ["--trust", str(kw["trust"])]
    _append_converge_to_cmd(cmd, kw.get("converge"))
    _append_verbose_to_cmd(cmd, kw.get("verbose", _VERBOSITY))

    if job == "ts":
        cmd += ["--transition", "yes", "--hessian", "first+last"]
    elif job == "irc":
        cmd += ["--irc", "yes"]
        cmd += ["--irc_direction", irc_direction]
        if hessian_path is not None:
            cmd += ["--hessian", f"file:{hessian_path}"]

    if prefix is not None:
        cmd += ["--prefix", prefix]

    _append_geometric_kwargs(cmd, kw, job=job)
    return cmd


def _load_ts_from_run_dir(run_dir: Path, GeoM, prefix: str):
    """Load optimized TS structure and energy from a completed TS run directory."""
    opt_path = run_dir / _prefix_optim_xyz(prefix)
    if not opt_path.is_file():
        cands = sorted(run_dir.glob("*optim*.xyz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            opt_path = cands[0]
        else:
            raise RuntimeError(f"No {_prefix_optim_xyz(prefix)} found in {run_dir}")
    loaded = GeoM(str(opt_path))
    ts_mol = loaded[-1] if len(loaded) > 0 else loaded
    log_path = run_dir / _prefix_log(prefix)
    energy = _parse_energy_hartree(ts_mol, log_path=log_path)
    return ts_mol, energy


def _load_irc_from_run_dir(run_dir: Path, GeoM, prefix: str):
    """Load IRC trajectory and per-frame energies from a completed IRC run directory.

    Returns
    -------
    (traj, energies) : (Molecule, list[float | None])
    """
    traj_name = _prefix_irc_traj(prefix)
    traj_path = run_dir / traj_name
    if not traj_path.is_file():
        cands = sorted(run_dir.glob("*_irc.xyz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            traj_path = cands[0]
        else:
            raise RuntimeError(f"No {traj_name} found in {run_dir}")

    traj = GeoM(str(traj_path))
    if not getattr(traj, "xyzs", None) or len(traj.xyzs) < 1:
        raise RuntimeError(f"IRC trajectory {traj_path} contains no frames.")

    log_path = run_dir / _prefix_log(prefix)
    comms = getattr(traj, "comms", None) or []
    qm_energies = getattr(traj, "qm_energies", None) or []

    energies = []
    for i in range(len(traj.xyzs)):
        comment = comms[i] if i < len(comms) else None
        energy = _parse_energy_hartree(traj, comment=comment, log_path=log_path)
        if energy is None and i < len(qm_energies):
            try:
                energy = float(qm_energies[i])
            except (TypeError, ValueError):
                pass
        energies.append(energy)

    return traj, energies


def _irc_endpoint_comm(existing_comm, energy, role: str) -> str:
    """Build an IRC endpoint comment, including energy when available."""
    if energy is not None:
        return f"IRC {role} Energy {energy:.8f}"
    if existing_comm:
        return existing_comm
    return f"IRC {role} endpoint"


def _neb_endpoints_molecule(chain, GeoM, energies=None):
    """Two-frame Molecule from the first and last NEB band images."""
    if not getattr(chain, "xyzs", None) or len(chain.xyzs) < 2:
        raise RuntimeError("NEB chain must contain at least 2 frames for endpoints.")
    endpoints = GeoM()
    endpoints.elem = list(chain.elem)
    endpoints.xyzs = [chain.xyzs[0].copy(), chain.xyzs[-1].copy()]
    comms = getattr(chain, "comms", None) or []
    endpoints.comms = [
        comms[0] if comms else "NEB reactant endpoint",
        comms[-1] if len(comms) >= len(chain.xyzs) else (comms[-1] if comms else "NEB product endpoint"),
    ]
    if energies:
        endpoints.qm_energies = [energies[0], energies[-1]]
    return endpoints


def _irc_endpoints_molecule(traj, GeoM, energies=None):
    """Two-frame Molecule from the first and last IRC images."""
    if not getattr(traj, "xyzs", None) or len(traj.xyzs) < 2:
        raise RuntimeError("IRC trajectory must contain at least 2 frames for endpoints.")
    endpoints = GeoM()
    endpoints.elem = list(traj.elem)
    comms = getattr(traj, "comms", None) or []
    endpoints.xyzs = [traj.xyzs[0].copy(), traj.xyzs[-1].copy()]

    re_comm = comms[0] if comms else None
    pr_comm = comms[-1] if len(comms) >= len(traj.xyzs) else (comms[0] if comms else None)
    re_e = energies[0] if energies else None
    pr_e = energies[-1] if energies else None

    endpoints.comms = [
        _irc_endpoint_comm(re_comm, re_e, "reactant"),
        _irc_endpoint_comm(pr_comm, pr_e, "product"),
    ]
    if energies:
        endpoints.qm_energies = [re_e, pr_e]
    return endpoints


def _verify_irc_endpoints_distinct(
    ep0,
    ep1,
    *,
    rmsd_threshold: float = 0.1,
    label: str = "IRC",
) -> None:
    """Raise if aligned RMSD between IRC endpoints is below *rmsd_threshold* (Å)."""
    rmsd = _drms_aligned(ep0, ep1)
    if rmsd < rmsd_threshold:
        raise WorkflowError(
            f"{label}: The two IRC endpoints are identical "
            f"(aligned RMSD = {rmsd:.4f} Å < {rmsd_threshold} Å). "
            "Reoptimize the TS structure."
        )


def _irc_result_dict(traj, energies, endpoints):
    return {"trj": traj, "energy": energies, "endpoints": endpoints}


def _copy_trajectory(mol, GeoM):
    """Shallow-copy a geomeTRIC Molecule trajectory (xyzs, comms, energies)."""
    out = GeoM()
    out.elem = list(mol.elem)
    out.xyzs = [xyz.copy() for xyz in mol.xyzs]
    comms = getattr(mol, "comms", None) or []
    out.comms = list(comms) if comms else [f"frame {i}" for i in range(len(out.xyzs))]
    qe = getattr(mol, "qm_energies", None)
    if qe:
        out.qm_energies = list(qe)
    return out


def _reverse_trajectory(mol, GeoM):
    """Return a new Molecule with frames, comments, and energies reversed."""
    out = GeoM()
    out.elem = list(mol.elem)
    out.xyzs = [xyz.copy() for xyz in reversed(mol.xyzs)]
    comms = getattr(mol, "comms", None) or []
    out.comms = list(reversed(comms)) if comms else [f"frame {i}" for i in range(len(out.xyzs))]
    qe = getattr(mol, "qm_energies", None)
    if qe:
        out.qm_energies = list(reversed(qe))
    return out


def _trajectory_energies(mol) -> List[Optional[float]]:
    """Per-frame energies from ``qm_energies`` or geomeTRIC-style comments."""
    n_frames = len(getattr(mol, "xyzs", None) or [])
    qe = getattr(mol, "qm_energies", None) or []
    comms = getattr(mol, "comms", None) or []
    energies: List[Optional[float]] = []
    for i in range(n_frames):
        if i < len(qe):
            try:
                energies.append(float(qe[i]))
                continue
            except (TypeError, ValueError):
                pass
        comment = comms[i] if i < len(comms) else None
        energies.append(_parse_energy_hartree(comment=comment))
    return energies


def _attach_trajectory_energies(mol, energies: Sequence[Optional[float]]) -> None:
    """Store per-frame energies on *mol* when any are available."""
    if any(e is not None for e in energies):
        mol.qm_energies = list(energies)


def _concat_trajectories(traj_a, traj_b, junction_xyz, rmsd_threshold: float, GeoM):
    """
    Concatenate two trajectories, optionally skipping duplicate junction frames.

    When the last frame of *traj_a* and the first frame of *traj_b* both align
    with *junction_xyz* within *rmsd_threshold*, the duplicate first frame of
    *traj_b* is dropped.
    """
    out = GeoM()
    out.elem = list(traj_a.elem)
    xyzs = [xyz.copy() for xyz in traj_a.xyzs]
    comms = list(
        getattr(traj_a, "comms", None) or [f"frame {i}" for i in range(len(traj_a.xyzs))]
    )
    energies = list(getattr(traj_a, "qm_energies", None) or _trajectory_energies(traj_a))

    start_b = 0
    if xyzs and _drms_aligned(xyzs[-1], junction_xyz) <= rmsd_threshold:
        if _drms_aligned(traj_b.xyzs[0], junction_xyz) <= rmsd_threshold:
            start_b = 1

    b_energies = getattr(traj_b, "qm_energies", None) or _trajectory_energies(traj_b)
    for i in range(start_b, len(traj_b.xyzs)):
        xyzs.append(traj_b.xyzs[i].copy())
        b_comms = getattr(traj_b, "comms", None) or []
        comms.append(b_comms[i] if i < len(b_comms) else f"frame {len(xyzs) - 1}")
        if i < len(b_energies):
            energies.append(b_energies[i])

    out.xyzs = xyzs
    out.comms = comms
    _attach_trajectory_energies(out, energies)
    return out


def _load_frame_opt_trajectory(frame_dir: Path, GeoM):
    """Load the full geomeTRIC optimization trajectory from a post-opt frame dir."""
    if not frame_dir.is_dir():
        raise RuntimeError(f"Post-opt frame directory not found: {frame_dir}")
    cands = sorted(frame_dir.glob("*optim*.xyz"))
    if not cands:
        raise RuntimeError(f"No optimization trajectory found in {frame_dir}")
    opt_path = cands[-1]
    traj = GeoM(str(opt_path))
    if not getattr(traj, "xyzs", None) or len(traj.xyzs) < 1:
        raise RuntimeError(f"Optimization trajectory {opt_path} contains no frames.")
    _attach_trajectory_energies(traj, _trajectory_energies(traj))
    return traj


def _build_postopt_irc_trajectory(
    reactant_opt,
    irc_traj,
    product_opt,
    irc_energies,
    rmsd_threshold: float,
    GeoM,
):
    """
    Build opt + IRC + opt trajectory for post-optimized IRC results.

    *reactant_opt* and *product_opt* are the full geomeTRIC optimization paths
    from ``<postopt>/frame_*/`` ``*_optim.xyz`` (IRC endpoint → minimum). Junction
    frames are dropped when they match the IRC endpoints within *rmsd_threshold*.

    The reactant optimization is reversed so the merged path runs
    reactant minimum → … → IRC reactant endpoint → IRC → IRC product endpoint
    → … → product minimum. Each IRC endpoint matches the starting frame of the
    corresponding ``*_optim.xyz`` trajectory.
    """
    reactant_part = _reverse_trajectory(reactant_opt, GeoM)
    irc_part = _copy_trajectory(irc_traj, GeoM)
    if irc_energies and len(irc_energies) == len(irc_part.xyzs):
        _attach_trajectory_energies(irc_part, irc_energies)
    else:
        _attach_trajectory_energies(irc_part, _trajectory_energies(irc_part))

    product_part = _copy_trajectory(product_opt, GeoM)
    _attach_trajectory_energies(product_part, _trajectory_energies(product_part))

    combined = _concat_trajectories(
        reactant_part,
        irc_part,
        irc_traj.xyzs[0],
        rmsd_threshold,
        GeoM,
    )
    combined = _concat_trajectories(
        combined,
        product_part,
        irc_traj.xyzs[-1],
        rmsd_threshold,
        GeoM,
    )
    return combined, _trajectory_energies(combined)


def _postoptimize_irc_endpoints(
    endpoints,
    *,
    base: Path,
    input_file: str,
    qm_program: str,
    nt: Optional[int],
    prefix: Optional[str],
    opt_kwargs: dict,
    GeoM,
):
    endpoints_xyz = base / "irc_endpoints.xyz"
    postopt_dir = base / "postopt"
    if postopt_dir.is_dir() and endpoints_xyz.is_file():
        try:
            cached_endpoints = GeoM(str(endpoints_xyz))
            if (
                len(getattr(cached_endpoints, "xyzs", []) or []) >= 2
                and np.allclose(cached_endpoints.xyzs[0], endpoints.xyzs[0], atol=1e-8)
                and np.allclose(cached_endpoints.xyzs[1], endpoints.xyzs[1], atol=1e-8)
            ):
                reactant_opt = _load_frame_opt_trajectory(postopt_dir / "frame_0", GeoM)
                product_opt = _load_frame_opt_trajectory(postopt_dir / "frame_1", GeoM)
                _log("Reusing cached IRC post-optimization", level=1)
                return cached_endpoints, reactant_opt, product_opt
        except Exception:
            pass
    endpoints.write(str(endpoints_xyz))
    kw = dict(opt_kwargs)
    if prefix is not None:
        kw["prefix"] = prefix
    _log(f"Post-optimizing IRC endpoints (run_dir={postopt_dir})...", level=1)
    optimized_endpoints = optimize_frames(
        input_file,
        str(endpoints_xyz),
        qm_program=qm_program,
        run_dir=str(postopt_dir),
        nt=nt,
        **kw,
    )
    reactant_opt = _load_frame_opt_trajectory(postopt_dir / "frame_0", GeoM)
    product_opt = _load_frame_opt_trajectory(postopt_dir / "frame_1", GeoM)
    return optimized_endpoints, reactant_opt, product_opt


def _package_irc_result(
    traj,
    energies,
    GeoM,
    *,
    postopt: bool,
    base: Path,
    input_file: str,
    qm_program: str,
    nt: Optional[int],
    prefix: Optional[str],
    irc_kwargs: dict,
):
    endpoints = _irc_endpoints_molecule(traj, GeoM, energies)
    rmsd_threshold = float(irc_kwargs.get("postopt_rmsd_threshold", 0.1))
    _verify_irc_endpoints_distinct(
        endpoints.xyzs[0],
        endpoints.xyzs[1],
        rmsd_threshold=rmsd_threshold,
        label=f"IRC in {base}",
    )
    result_traj = traj
    result_energies = energies
    if postopt:
        postopt_kw = {
            k: irc_kwargs[k]
            for k in (
                "coordsys", "maxiter", "timeout", "converge",
                "postopt_converge", "verbose",
            )
            if k in irc_kwargs
        }
        if "postopt_converge" in postopt_kw:
            postopt_kw["converge"] = postopt_kw.pop("postopt_converge")
        endpoints, reactant_opt, product_opt = _postoptimize_irc_endpoints(
            endpoints,
            base=base,
            input_file=input_file,
            qm_program=qm_program,
            nt=nt,
            prefix=prefix,
            opt_kwargs=postopt_kw,
            GeoM=GeoM,
        )
        _verify_irc_endpoints_distinct(
            endpoints.xyzs[0],
            endpoints.xyzs[1],
            rmsd_threshold=rmsd_threshold,
            label=f"IRC in {base}",
        )
        result_traj, result_energies = _build_postopt_irc_trajectory(
            reactant_opt,
            traj,
            product_opt,
            energies,
            rmsd_threshold,
            GeoM,
        )
        artifact_prefix = _output_prefix(base / Path(input_file).name, prefix)
        postopt_path = _write_postopt_irc_trajectory(result_traj, base, artifact_prefix)
        _log(
            f"Post-opt IRC trajectory: {len(result_traj.xyzs)} frames "
            f"({len(reactant_opt.xyzs)}-frame reactant *_optim.xyz + "
            f"{len(traj.xyzs)} IRC + {len(product_opt.xyzs)}-frame product "
            f"*_optim.xyz; junctions at IRC endpoints) → {postopt_path.name}",
            level=0,
        )
    return _irc_result_dict(result_traj, result_energies, endpoints)


def optimize_frames(input_file, xyz_file, *, qm_program="psi4", run_dir=None, **opt_kwargs):
    """
    Optimize one or more geometries using geomeTRIC.

    Provide a QM input template via ``input_file`` (method, basis, charge,
    multiplicity, etc.). Geometry comes from ``xyz_file`` (single- or
    multi-frame).

    Example usage:

        opt_mols = optimize_frames(
            "psi4.in",
            "initial.xyz",
            qm_program="psi4",
            run_dir="opt_runs",
            nt=4,
        )

    A single-frame XYZ is valid; only ``frame_0/`` is created under ``run_dir``.

    Internally this runs something equivalent to:

        geometric-optimize <prepared_input> --engine psi4 --coords <frame.xyz> ...

    Parameters
    ----------
    input_file : str
        Path to the QM input template file (e.g. psi4.in).
    xyz_file : str
        Path to an XYZ file with one or more frames to optimize.
    qm_program : str
        QM program / engine to use (e.g. "psi4", "qchem", ...).
        Default: "psi4".
    run_dir : str or Path or None
        Parent directory for per-frame subdirs (``frame_0/``, ``frame_1/``, ...).
        When given, used as-is. Otherwise defaults to the process cwd.
    prefix : str, optional
        geomeTRIC ``--prefix`` for log/artifact basenames. When omitted, geomeTRIC
        uses the input-file stem (e.g. ``psi4`` from ``psi4.in``).
    **opt_kwargs
        Passed to geomeTRIC (e.g. coordsys, maxiter, nt, timeout, prefix, ...).

    Returns
    -------
    OptMs : geometric.molecule.Molecule
        A multi-frame Molecule containing the optimized structures (in .xyzs),
        comments, and .qm_energies (if parseable from comments or qm_energies).
    """

    if not input_file:
        raise ValueError("optimize_frames requires `input_file` (QM input template).")
    if not xyz_file:
        raise ValueError("optimize_frames requires `xyz_file` (path to XYZ, possibly multi-frame).")

    verbose = _pop_verbose(opt_kwargs)

    import shutil
    import subprocess
    import re
    from pathlib import Path

    GeoM = _get_geo_molecule()

    user_input = Path(input_file)
    xyz_input = Path(xyz_file)
    if not user_input.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not xyz_input.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")

    M = GeoM(str(xyz_input))
    n_frames = len(M.xyzs) if hasattr(M, 'xyzs') and M.xyzs else 0
    if n_frames == 0:
        # return empty
        OptMs = GeoM()
        if hasattr(M, 'elem') and M.elem:
            OptMs.elem = list(M.elem)
        return OptMs

    # Use list accumulation + final fresh Molecule construction.
    # This completely sidesteps __iadd__ on empty objects and Data[key][0] assumptions.
    collected_xyzs = []
    collected_comms = []
    energies = []

    user_prefix = opt_kwargs.pop("prefix", None)
    base_for_run = _resolve_run_dir(run_dir)
    opt_cmd_kwargs = {**opt_kwargs, "verbose": verbose}

    try:
        for frame_num in range(n_frames):
            current_run_dir = base_for_run / f"frame_{frame_num}"
            coords_xyz = current_run_dir / "coords.xyz"
            dest_input = current_run_dir / user_input.name
            prefix = _output_prefix(dest_input, user_prefix)
            opt_path = current_run_dir / _prefix_optim_xyz(prefix)

            # Build the command we intend to run (for cache command comparison).
            # We do this early so the cache logic can compare against the logged command.
            cmd_input = _geometric_argv_path(dest_input, current_run_dir)
            cmd_coords = _geometric_argv_path(coords_xyz, current_run_dir)
            cmd_list = _build_geometric_cmd(
                "opt",
                cmd_input,
                cmd_coords,
                qm_program,
                opt_cmd_kwargs,
                prefix=user_prefix,
            )
            intended_cmd_str = " ".join(cmd_list)

            reused = False
            if (current_run_dir.exists()
                    and coords_xyz.exists()
                    and dest_input.exists()
                    and opt_path.exists()):
                try:
                    cached_m = GeoM(str(coords_xyz))
                    if cached_m.xyzs and len(cached_m.xyzs) > 0:
                        if np.allclose(M.xyzs[frame_num], cached_m.xyzs[0], atol=1e-8):
                            preferred_log = current_run_dir / _prefix_log(prefix)
                            log_path = (
                                preferred_log
                                if preferred_log.is_file()
                                else _find_converged_log_in_dir(
                                    current_run_dir,
                                    preferred=preferred_log,
                                )
                            )
                            artifact_hit = _optimization_artifact_cache_hit(
                                current_run_dir,
                                user_input,
                                M.xyzs[frame_num],
                                prefix,
                                GeoM,
                                qm_program=qm_program,
                            )
                            decision = _converged_cache_decision(
                                log_path=log_path,
                                cached_input=dest_input,
                                current_input=user_input,
                                prev_cmd_str=_read_logged_geometric_cmd(
                                    log_path,
                                    run_dir=current_run_dir,
                                ),
                                intended_cmd_str=intended_cmd_str,
                                label=f"frame {frame_num}",
                                qm_program=qm_program,
                                allow_artifact_fallback=True,
                                artifact_ok=artifact_hit,
                            )
                            if decision == "invalidate":
                                _rename_conflicting_cache_dir(current_run_dir)
                            elif decision == "fail":
                                _raise_failed_cache(
                                    f"Optimization frame {frame_num} in {current_run_dir}",
                                    log_path,
                                )
                            elif decision == "reuse":
                                if log_path is None:
                                    _log(
                                        f"Reusing optimization artifacts for frame {frame_num} "
                                        f"from {current_run_dir}",
                                        level=1,
                                    )
                                loaded = GeoM(str(opt_path))
                                opt_mol = loaded[-1] if len(loaded) > 0 else loaded

                                if opt_mol.xyzs:
                                    collected_xyzs.append(opt_mol.xyzs[0].copy())
                                else:
                                    collected_xyzs.append(M.xyzs[frame_num].copy())

                                cmm = (opt_mol.comms[0]
                                       if getattr(opt_mol, "comms", None)
                                       else f"cached frame {frame_num}")
                                collected_comms.append(cmm)

                                energy = _parse_energy_hartree(
                                    opt_mol,
                                    comment=cmm,
                                    log_path=log_path or (current_run_dir / _prefix_log(prefix)),
                                )
                                energies.append(energy)

                                _log(
                                    f"Reusing cached result for frame {frame_num} from {current_run_dir}",
                                    level=1,
                                )
                                reused = True
                            elif log_path is None and not artifact_hit:
                                _log(
                                    f"No converged optimization log or artifacts for frame {frame_num}; "
                                    "will re-optimize.",
                                    level=1,
                                )
                except Exception as cache_err:
                    _log(f"Cache check failed for frame {frame_num} ({cache_err}); will re-optimize.", level=1)

            if reused:
                continue

            # No usable cache — perform the optimization (or re-optimization).
            current_run_dir.mkdir(parents=True, exist_ok=True)

            # Copy template (never mutate user's original)
            shutil.copy(user_input, dest_input)

            # Fresh single-frame Molecule for the coords file (avoids state corruption
            # from slicing/iterating the source multi-frame M).
            single = GeoM()
            single.elem = list(M.elem) if hasattr(M, 'elem') else []
            single.xyzs = [M.xyzs[frame_num].copy()]
            if hasattr(M, 'comms') and M.comms and len(M.comms) > frame_num:
                single.comms = [M.comms[frame_num]]
            else:
                single.comms = [f"Frame {frame_num} from {xyz_input.name}"]

            coords_xyz = current_run_dir / "coords.xyz"
            single.write(str(coords_xyz))

            cwd = current_run_dir
            input_for_geo = dest_input

            # Command: geometric-optimize <template> --engine <qm> --coords <theframecoords> ...
            base_cmd = _build_geometric_cmd(
                "opt",
                _geometric_argv_path(input_for_geo, cwd),
                _geometric_argv_path(coords_xyz, cwd),
                qm_program,
                opt_cmd_kwargs,
                prefix=user_prefix,
            )

            _log(f"Running: {base_cmd}", level=1)
            _log(f"Working directory: {cwd}", level=1)
            _write_last_command(cwd, base_cmd)

            result = subprocess.run(
                base_cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                **_subprocess_timeout_kwargs(opt_kwargs),
            )

            # Load the result even on non-zero return (geometric returns non-zero on
            # "max iterations reached" or other non-fatal "did not fully converge").
            # We accept the last geometry present in the output trajectory as the
            # best available result for this frame (consistent with partial-result
            # tolerance used in tests and higher-level workflows).
            if result.returncode != 0:
                _log_warn("geometric-optimize returned non-zero (may be maxiter or converge criteria). stderr tail:")
                _log_warn(result.stderr[-1500:] if result.stderr else "(no stderr)")

            # Load the result. geometric's <prefix>_optim.xyz often contains the
            # optimization trajectory; we take the last frame as the optimized structure.
            opt_path = cwd / _prefix_optim_xyz(prefix)
            if not opt_path.exists():
                # be flexible with output names
                cands = sorted(cwd.glob("*optim*.xyz"))
                if cands:
                    opt_path = cands[-1]
                else:
                    _log_warn("No optimized .xyz output found; hard failure for this frame.")
                    raise RuntimeError(f"No optimized .xyz output found in {cwd}")

            loaded = GeoM(str(opt_path))
            opt_mol = loaded[-1] if len(loaded) > 0 else loaded

            # Collect the (single) optimized frame data
            if opt_mol.xyzs:
                collected_xyzs.append(opt_mol.xyzs[0].copy())
            else:
                # fallback, should not happen
                collected_xyzs.append(M.xyzs[frame_num].copy())

            cmm = opt_mol.comms[0] if getattr(opt_mol, "comms", None) else f"optimized frame {frame_num}"
            collected_comms.append(cmm)

            combined_log = (result.stdout or "") + "\n" + (result.stderr or "")
            energy = _parse_energy_hartree(
                opt_mol,
                comment=cmm,
                log_path=cwd / _prefix_log(prefix),
                combined_log=combined_log,
            )
            energies.append(energy)

            if energy is not None:
                _log(f"Optimization complete. Energy = {energy:.8f} Ha", level=1)
            else:
                _log("Optimization complete (energy not parsed)", level=1)

            # Leave per-frame dirs in place (user can inspect; no auto-clean when using run_dir or cwd)

        # Build final multi-frame result with a fresh Molecule + direct list assignment.
        OptMs = GeoM()
        if hasattr(M, 'elem') and M.elem:
            OptMs.elem = list(M.elem)
        OptMs.xyzs = collected_xyzs
        OptMs.comms = collected_comms
        # Attach energies for downstream convenience (len matches ns)
        if energies:
            OptMs.qm_energies = energies

        return OptMs

    except Exception as e:
        _log_warn(f"optimize_frames failed: {e}")
        OptMs = GeoM()
        if 'M' in locals() and hasattr(M, 'elem') and M.elem:
            OptMs.elem = list(M.elem)
        return OptMs


# =============================================================================
# Elementary Step Refinement (Direct/Local Mode)
# NEB → TS Optimization → IRC
# =============================================================================

def _load_neb_converged_chain(run_dir: Path, GeoM, prefix: str):
    """Load the final NEB band from the highest-numbered <prefix>.tmp/chain_*.xyz."""
    chain_dir = run_dir / f"{prefix}.tmp"
    if not chain_dir.is_dir():
        return None
    chains = sorted(
        chain_dir.glob("chain_*.xyz"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not chains:
        return None
    return GeoM(str(chains[-1]))


def run_neb(
    input_file: str,
    initial_chain: Union[str, Path],
    *,
    qm_program: str = "psi4",
    n_images: int = 11,
    spring_constant: float = 1.0,
    nt: Optional[int] = None,
    run_dir: Optional[str] = None,
    **neb_kwargs,
):
    """
    Run geomeTRIC NEB from a multi-frame initial-chain XYZ and a QM input template.

    Parameters
    ----------
    input_file : str
        QM program input template (geometry in the file is ignored; chain coords come
        from ``initial_chain``).
    initial_chain : str or Path
        XYZ file containing the full NEB initial path (all images).
    qm_program : str
        geomeTRIC engine name (e.g. ``psi4``).
    run_dir : str or Path, optional
        Working directory for the NEB run. When given, used as-is. Otherwise
        defaults to the process cwd.
    prefix : str, optional
        geomeTRIC ``--prefix`` for log/artifact basenames. When omitted, geomeTRIC
        uses the input-file stem (e.g. ``psi4`` from ``psi4.in``).

    Copies ``input_file`` and ``initial_chain`` into the run directory using their
    original basenames, then invokes ``geometric-neb <input> <chain.xyz> ...`` (or
    ``python -m geometric.optimize`` with ``--neb``) with ``cwd`` set to that directory.

    Supports caching when the chain + template match, the log contains a recognized
    convergence marker (see ``_check_log_for_caching``), and the command line matches.

    Expects geomeTRIC to write the converged band in ``<prefix>.tmp/chain_NNNN.xyz``.
    When a barrier is present, ``<prefix>.tsClimb.xyz`` is also produced; when no
    climbing image is found, ``ts_guess`` is ``None`` and only the converged chain
    is returned.

    Returns
    -------
    dict
        ``optimized_chain`` : converged NEB band (``Molecule``, energies in
        ``qm_energies``)
        ``ts_guess`` : climbing-image structure from ``<prefix>.tsClimb.xyz``, or
        ``None`` when no climbing image was detected
    """
    import subprocess
    import shutil
    from pathlib import Path

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for NEB in direct mode.")

    chain_src = Path(initial_chain)
    if not chain_src.is_file():
        raise FileNotFoundError(f"NEB initial chain not found: {initial_chain}")

    qm_program = neb_kwargs.pop("qm_program", None) or qm_program or "psi4"
    inp_path = neb_kwargs.pop("input_file", None) or input_file
    if not inp_path:
        raise ValueError("run_neb requires input_file (QM template).")
    inp_src = Path(inp_path)
    if not inp_src.is_file():
        raise FileNotFoundError(f"NEB input file not found: {inp_path}")

    chain_mol = GeoM(str(chain_src))

    user_prefix = neb_kwargs.pop("prefix", None)
    tmp = _resolve_run_dir(run_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    verbose = _pop_verbose(neb_kwargs)
    neb_cmd_kwargs = {**neb_kwargs, "verbose": verbose}
    _log(f"Running NEB from {chain_src.name} ({n_images} images)...", level=1)

    chain_dest = tmp / chain_src.name
    inp_dest = tmp / inp_src.name
    prefix = _output_prefix(inp_dest, user_prefix)

    intended_cmd_list = _build_geometric_cmd(
        "neb",
        _geometric_argv_path(inp_dest, tmp),
        _geometric_argv_path(chain_dest, tmp),
        qm_program,
        neb_cmd_kwargs,
        nt=nt,
        prefix=user_prefix,
        n_images=n_images,
        spring_constant=spring_constant,
    )
    intended_cmd_str = " ".join(intended_cmd_list)

    log_path = tmp / _prefix_log(prefix)
    ts_climb_path = tmp / _prefix_neb_ts_climb(prefix)

    cached = False
    if chain_dest.exists() and inp_dest.exists():
        try:
            pm = GeoM(str(chain_dest))
            if len(pm.xyzs) == len(chain_mol.xyzs) and all(
                np.allclose(p, c, atol=1e-8) for p, c in zip(pm.xyzs, chain_mol.xyzs)
            ):
                fc_cached = _load_neb_converged_chain(tmp, GeoM, prefix)
                if fc_cached is not None:
                    prev_cmd_str = _read_logged_geometric_cmd(log_path, run_dir=tmp)
                    decision = _converged_cache_decision(
                        log_path=log_path,
                        cached_input=inp_dest,
                        current_input=inp_src,
                        prev_cmd_str=prev_cmd_str,
                        intended_cmd_str=intended_cmd_str,
                        label=f"NEB in {tmp}",
                        qm_program=qm_program,
                    )
                    if decision == "invalidate":
                        _rename_conflicting_cache_dir(tmp)
                    elif decision == "fail":
                        _raise_failed_cache(f"NEB in {tmp}", log_path)
                    elif decision == "reuse":
                        _log(f"Reusing cached NEB result from {tmp}", level=1)
                        cached = True
                    elif not _check_log_for_caching(log_path):
                        _log(f"{log_path.name} shows no convergence marker; will re-run NEB.", level=1)
        except Exception as ce:
            _log(f"NEB cache check error ({ce}); will re-run.", level=1)

    if not cached:
        shutil.copy2(chain_src, chain_dest)
        shutil.copy2(inp_src, inp_dest)

        cmd = _build_geometric_cmd(
            "neb",
            _geometric_argv_path(inp_dest, tmp),
            _geometric_argv_path(chain_dest, tmp),
            qm_program,
            neb_cmd_kwargs,
            nt=nt,
            prefix=user_prefix,
            n_images=n_images,
            spring_constant=spring_constant,
        )

        _write_last_command(tmp, cmd)

        _log(f"Running: {cmd}", level=1)
        _log(f"Working directory: {tmp}", level=1)

        res = subprocess.run(
            cmd,
            cwd=str(tmp),
            capture_output=True,
            text=True,
            **_subprocess_timeout_kwargs(neb_kwargs),
        )
        if res.returncode != 0:
            _log_warn(f"geometric NEB failed (engine={qm_program}).")
            _log_warn((res.stderr or "")[-2000:])
            raise RuntimeError(f"NEB calculation failed (engine={qm_program}). Check {log_path}.")

    fc = _load_neb_converged_chain(tmp, GeoM, prefix)
    if fc is None:
        raise RuntimeError(
            f"NEB did not produce a converged band under {tmp / (prefix + '.tmp')}. "
            f"Check {log_path} for details."
        )
    nn = len(fc)

    energies = []
    for i in range(nn):
        cmm = fc.comms[i] if getattr(fc, "comms", None) and i < len(fc.comms) else None
        energies.append(_parse_energy_hartree(fc, comment=cmm))
    fc.qm_energies = energies

    ts_guess = GeoM(str(ts_climb_path)) if ts_climb_path.is_file() else None

    return {"optimized_chain": fc, "ts_guess": ts_guess}


def _frames_to_molecule(frames, GeoM):
    """Pack single- or multi-frame inputs into one geomeTRIC Molecule."""
    if not frames:
        raise ValueError("At least one structure is required.")
    batch = GeoM()
    ref = frames[0]
    batch.elem = list(getattr(ref, "elem", []) or [])
    xyzs = []
    comms = []
    for i, frame in enumerate(frames):
        if hasattr(frame, "xyzs") and frame.xyzs:
            xyzs.append(frame.xyzs[0].copy())
            frame_comms = getattr(frame, "comms", None) or []
            comms.append(frame_comms[0] if frame_comms else f"frame {i}")
        else:
            xyzs.append(np.asarray(frame, dtype=float).copy())
            comms.append(f"frame {i}")
    batch.xyzs = xyzs
    batch.comms = comms
    return batch


_INTERP_CANONICAL_NAME = "interpolated.xyz"
_TRICS_INTERP_PREFERRED_NAMES = (
    "interpolated_TRICS_prealigned.xyz",
    "interpolated_TRICS.xyz",
)
_INTERP_LEGACY_OUTPUT_NAMES = (
    "interpolated_splice.xyz",
    "interpolated_fast_mode.xyz",
    "initial_guess.xyz",
)


def _find_trics_interpolation_xyz(run_dir: Path) -> Optional[Path]:
    """Return a geomeTRIC TRICS interpolation XYZ in *run_dir*, if present."""
    for name in _TRICS_INTERP_PREFERRED_NAMES:
        path = run_dir / name
        if path.is_file():
            return path
    candidates = [
        p
        for p in run_dir.glob("*interpolated_TRICS*.xyz")
        if p.name != "interpolated_TRICS_extra.xyz"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _promote_trics_interpolation_to_canonical(run_dir: Path) -> Optional[Path]:
    """Copy ``*interpolated_TRICS*.xyz`` to ``interpolated.xyz`` when needed."""
    import shutil

    canonical = run_dir / _INTERP_CANONICAL_NAME
    if canonical.is_file():
        return canonical
    src = _find_trics_interpolation_xyz(run_dir)
    if src is None:
        return None
    shutil.copy2(src, canonical)
    return canonical


def _find_interpolation_output(run_dir: Path) -> Optional[Path]:
    """Return the usable interpolation XYZ in *run_dir*, if any."""
    canonical = run_dir / _INTERP_CANONICAL_NAME
    if canonical.is_file():
        return canonical
    trics = _find_trics_interpolation_xyz(run_dir)
    if trics is not None:
        return trics
    candidates = [
        run_dir / name
        for name in _INTERP_LEGACY_OUTPUT_NAMES
        if (run_dir / name).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_interpolation_output(run_dir: Path, GeoM):
    """Load the interpolated path written by geomeTRIC in *run_dir*."""
    out_path = _find_interpolation_output(run_dir)
    if out_path is None:
        raise RuntimeError(
            f"geometric-interpolate did not produce an output XYZ in {run_dir}. "
            f"Check {run_dir / 'interpolate.log'} for details."
        )
    return GeoM(str(out_path))


def _two_frame_xyz_matches(path: Path, start_xyz, end_xyz, GeoM, atol: float = 1e-8) -> bool:
    """True when *path* is a two-frame XYZ matching the given endpoint coordinates."""
    if not path.is_file():
        return False
    try:
        mol = GeoM(str(path))
    except Exception:
        return False
    if len(getattr(mol, "xyzs", []) or []) < 2:
        return False
    return (
        np.allclose(mol.xyzs[0], np.asarray(start_xyz, dtype=float), atol=atol)
        and np.allclose(mol.xyzs[-1], np.asarray(end_xyz, dtype=float), atol=atol)
    )


def _interpolation_request_key(
    n_images: int,
    prefix: str,
    **kwargs,
) -> str:
    payload = {"n_images": int(n_images), "prefix": str(prefix)}
    payload.update(kwargs)
    return json.dumps(payload, sort_keys=True)


def _read_interpolation_request(run_dir: Path) -> Optional[str]:
    path = run_dir / "last_interpolation_request.txt"
    if not path.is_file():
        return None
    return path.read_text().strip() or None


def _write_interpolation_request(run_dir: Path, request_key: str) -> None:
    (run_dir / "last_interpolation_request.txt").write_text(request_key)


def _interpolation_cache_hit(
    run_dir: Path,
    endpoints_path: Path,
    n_images: int,
    GeoM,
    *,
    log_prefix: str = "interpolate",
) -> bool:
    """True when a converged interpolation output exists for the current endpoints file."""
    return _interpolation_cache_decision(
        run_dir,
        endpoints_path,
        n_images,
        GeoM,
        log_prefix=log_prefix,
        request_key=None,
    ) == "reuse"


def _interpolation_cache_decision(
    run_dir: Path,
    endpoints_path: Path,
    n_images: int,
    GeoM,
    *,
    log_prefix: str = "interpolate",
    request_key: Optional[str],
) -> str:
    """
    Decide whether a prior TRICS interpolation may be reused.

    Returns ``\"reuse\"``, ``\"rerun\"``, or ``\"fail\"``.
    """
    if not endpoints_path.is_file():
        return "rerun"
    log_path = run_dir / f"{log_prefix}.log"
    out_path = _find_interpolation_output(run_dir)
    if (
        _check_log_for_caching(log_path, markers=("Converged!",))
        and out_path is not None
    ):
        try:
            mol = GeoM(str(out_path))
            n_frames = len(getattr(mol, "xyzs", []) or [])
        except Exception:
            n_frames = 0
        if n_frames >= 5 and n_frames == n_images:
            return "reuse"
    if (
        request_key is not None
        and log_path.is_file()
        and not _check_log_for_caching(log_path, markers=("Converged!",))
        and _read_interpolation_request(run_dir) == request_key
    ):
        return "fail"
    return "rerun"


def _ensure_canonical_interpolated_xyz(canonical: Path, step_dir: Path) -> Path:
    """Ensure ``interpolated.xyz`` exists, copying from TRICS or legacy outputs if needed."""
    import shutil

    if canonical.is_file():
        return canonical
    promoted = _promote_trics_interpolation_to_canonical(step_dir)
    if promoted is not None and promoted.is_file():
        if promoted.resolve() == canonical.resolve():
            return canonical
        shutil.copy2(promoted, canonical)
        return canonical
    src = _find_interpolation_output(step_dir)
    if src is None:
        raise RuntimeError(f"No interpolation output found in {step_dir}")
    shutil.copy2(src, canonical)
    return canonical


def _run_trics_interpolation(
    endpoints_xyz: Union[str, Path],
    *,
    n_images: int = 50,
    prefix: Optional[str] = None,
    run_dir: Optional[Union[str, Path]] = None,
    **kwargs,
):
    """Run geomeTRIC TRICS interpolation in-process (logs to ``<prefix>.log`` only)."""
    import os
    import shutil

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for TRICS interpolation.")
    try:
        from geometric.config import config_dir
        from geometric.interpolate import run_interpolator
    except ImportError as exc:
        raise RuntimeError(
            "geomeTRIC TRICS interpolation requires geometric.interpolate "
            "(install the geomeTRIC interpolate branch)."
        ) from exc

    endpoints_src = Path(endpoints_xyz)
    if not endpoints_src.is_file():
        raise FileNotFoundError(f"Interpolation endpoints not found: {endpoints_xyz}")

    run_dir = _resolve_run_dir(run_dir or endpoints_src.parent)
    run_dir.mkdir(parents=True, exist_ok=True)

    endpoints_dest = run_dir / endpoints_src.name
    if not endpoints_dest.exists() or endpoints_src.resolve() != endpoints_dest.resolve():
        shutil.copy2(endpoints_src, endpoints_dest)

    log_prefix = kwargs.pop("log_prefix", None)
    prefix = log_prefix if log_prefix is not None else prefix
    if prefix is None:
        prefix = "interpolate"

    n_images = max(int(kwargs.pop("n_images", n_images)), 5)
    for key in ("timeout", "block_until_ms"):
        kwargs.pop(key, None)

    log_path = Path(f"{prefix}.log") if not str(prefix).endswith(".log") else Path(prefix)
    if not log_path.is_absolute():
        log_path = run_dir / log_path.name

    log_ini = os.path.join(config_dir, "logFile.ini")
    if not os.path.isfile(log_ini):
        raise FileNotFoundError(
            f"geomeTRIC logging config not found: {log_ini}. "
            "Reinstall geomeTRIC from the interpolate branch."
        )

    request_key = _interpolation_request_key(
        n_images,
        str(prefix),
        fast=kwargs.get("fast", False),
        align_system=kwargs.get("align_system", True),
        align_frags=kwargs.get("align_frags", False),
        verbose=kwargs.get("verbose", 0),
        extrapolate=kwargs.get("extrapolate"),
    )
    interp_log_path = run_dir / (
        f"{prefix}.log" if not str(prefix).endswith(".log") else Path(prefix).name
    )
    interp_decision = _interpolation_cache_decision(
        run_dir,
        endpoints_dest,
        n_images,
        GeoM,
        log_prefix=str(prefix),
        request_key=request_key,
    )
    if interp_decision == "reuse":
        _log("Reusing cached TRICS interpolation", level=1)
        return _load_interpolation_output(run_dir, GeoM)
    if interp_decision == "fail":
        _raise_failed_cache(f"TRICS interpolation in {run_dir}", interp_log_path)

    interp_kwargs = {
        "input": endpoints_dest.name,
        "prefix": str(prefix),
        "nframes": n_images,
        "logIni": log_ini,
        "fast": kwargs.pop("fast", False),
        "align_system": kwargs.pop("align_system", True),
        "align_frags": kwargs.pop("align_frags", False),
        "verbose": kwargs.pop("verbose", 0),
    }
    if "extrapolate" in kwargs:
        interp_kwargs["extrapolate"] = kwargs.pop("extrapolate")

    _log(f"TRICS interpolation running in {run_dir}...", level=1)
    _write_interpolation_request(run_dir, request_key)

    prev_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        run_interpolator(**interp_kwargs)
    except Exception:
        _log_warn(f"TRICS interpolation failed. Check {log_path} for details.")
        raise
    finally:
        os.chdir(prev_cwd)

    _promote_trics_interpolation_to_canonical(run_dir)
    out = _load_interpolation_output(run_dir, GeoM)
    _log(f"TRICS interpolation complete ({len(out.xyzs)} frames).", level=1)
    return out


def interpolate(
    endpoints: Union[str, Path, Sequence, "GeoMolecule"],
    n_images: int = 50,
    **kwargs,
):
    """
    TRICS interpolation between endpoint structures.

    Accepts either a multi-frame XYZ path or in-memory endpoint geometries.
    This is the interpolation backend used by :class:`TRICWorkflow`.

    Parameters
    ----------
    endpoints : str, Path, sequence, or Molecule
        Path to a multi-frame XYZ file, a multi-frame geomeTRIC ``Molecule``,
        or a sequence of single-frame ``Molecule`` objects / coordinate arrays.
    n_images : int
        Number of interpolated images (minimum 5).
    run_dir : str or Path, optional
        Working directory for interpolation artifacts. When *endpoints* are
        passed in memory and ``run_dir`` is set, ``endpoints.xyz`` is written
        there; otherwise a temporary directory is used.
    fast : bool
        Use geomeTRIC fast mode (endpoint-to-endpoint TRICS path, no splicing).
        Default ``False``.
    align_system : bool
        Align reactant/product before interpolating. Default ``True``.
    align_frags : bool
        Pre-align fragments before interpolating. Default ``False``.

    Returns
    -------
    Molecule
        Interpolated TRICS pathway.
    """
    if isinstance(endpoints, (str, Path)):
        path = Path(endpoints)
        if not path.is_file():
            raise FileNotFoundError(f"Interpolation endpoints not found: {endpoints}")
        return _run_trics_interpolation(path, n_images=n_images, **kwargs)

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for TRICS interpolation.")

    if hasattr(endpoints, "xyzs") and len(getattr(endpoints, "xyzs", []) or []) >= 2:
        mol = endpoints
    else:
        mol = _frames_to_molecule(endpoints, GeoM)

    import tempfile

    run_dir = kwargs.get("run_dir")
    if run_dir is not None:
        step_dir = Path(run_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        endpoints_path = step_dir / "endpoints.xyz"
        mol.write(str(endpoints_path))
        return _run_trics_interpolation(endpoints_path, n_images=n_images, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        endpoints_path = Path(tmpdir) / "endpoints.xyz"
        mol.write(str(endpoints_path))
        return _run_trics_interpolation(endpoints_path, n_images=n_images, **kwargs)


def optimize_ts(
    input_file: str,
    xyz_file: str,
    *,
    qm_program: str = "psi4",
    nt: Optional[int] = None,
    run_dir: Optional[str] = None,
    **opt_kwargs,
):
    """
    Optimize a transition state guess (typically the highest-energy image from NEB,
    i.e. a *.tsClimb.xyz file) using geomeTRIC driven by a user QM input template.

    This follows the same single-frame path as ``optimize_frames`` (template copy,
    ``coords.xyz`` in ``run_dir``, cache checks), but always passes TS-specific flags:

        --transition yes
        --hessian first+last

    Recommended usage after run_neb (when using run_dir="neb_run"):

        ts_climb = sorted(Path("neb_run").glob("*tsClimb*.xyz"))[-1]
        ts_mol, energy = optimize_ts(
            "psi4.in",
            str(ts_climb),
            qm_program="psi4",
            nt=2,
            run_dir="ts_run",
        )

    Parameters
    ----------
    input_file : str
        QM input template (e.g. psi4.in). Geometry is supplied via ``--coords``.
    xyz_file : str or Path
        Path to the TS guess coordinates (single- or multi-frame; multi-frame
        uses the last frame, matching NEB tsClimb convention).
    qm_program : str
        QM engine (default ``psi4``).
    run_dir : str or Path, optional
        Working directory for this TS optimization. Defaults to the process cwd.
    prefix : str, optional
        geomeTRIC ``--prefix`` for log/artifact basenames.
    nt : int, optional
        Number of threads / MPI ranks to pass as ``--nt``.
    **opt_kwargs
        Forwarded (e.g. maxiter, coordsys, timeout, ...).

    Returns
    -------
    (ts_mol, energy) : (Molecule, float or None)
        The optimized TS structure (last frame of the optim xyz) and the energy.
    """
    if not input_file:
        raise ValueError("optimize_ts requires `input_file` (QM template).")
    if not xyz_file:
        raise ValueError("optimize_ts requires `xyz_file` (e.g. the *.tsClimb.xyz from NEB).")

    verbose = _pop_verbose(opt_kwargs)
    ts_cmd_kwargs = {**opt_kwargs, "verbose": verbose}
    import shutil
    import subprocess
    from pathlib import Path

    _log(f"Running TS optimization from {Path(xyz_file).name}", level=1)

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for optimize_ts in direct mode.")

    qm_program = opt_kwargs.pop("qm_program", None) or qm_program or "psi4"
    inp_file = opt_kwargs.pop("input_file", None) or input_file

    user_input = Path(inp_file)
    xyz_input = Path(xyz_file)
    if not user_input.exists():
        raise FileNotFoundError(f"Input file not found: {inp_file}")
    if not xyz_input.exists():
        raise FileNotFoundError(f"xyz_file not found: {xyz_file}")

    M = GeoM(str(xyz_input))
    if not getattr(M, "xyzs", None):
        raise ValueError(f"xyz_file contains no coordinates: {xyz_file}")
    frame_num = (len(M.xyzs) - 1) if len(M.xyzs) > 1 else 0
    src_xyz = M.xyzs[frame_num]

    user_prefix = opt_kwargs.pop("prefix", None)
    base = _resolve_run_dir(run_dir)

    dest_input = base / user_input.name
    prefix = _output_prefix(dest_input, user_prefix)
    log_path = base / _prefix_log(prefix)
    opt_path = base / _prefix_optim_xyz(prefix)
    coords_xyz = base / "coords.xyz"

    intended_cmd_list = _build_geometric_cmd(
        "ts",
        _geometric_argv_path(dest_input, base),
        _geometric_argv_path(coords_xyz, base),
        qm_program,
        ts_cmd_kwargs,
        nt=nt,
        prefix=user_prefix,
    )
    intended_cmd_str = " ".join(intended_cmd_list)

    reused = False
    if (base.exists()
            and dest_input.exists()
            and coords_xyz.exists()
            and opt_path.exists()
            and log_path.exists()):
        try:
            cached_m = GeoM(str(coords_xyz))
            if cached_m.xyzs and np.allclose(cached_m.xyzs[0], src_xyz, atol=1e-8):
                decision = _converged_cache_decision(
                    log_path=log_path,
                    cached_input=dest_input,
                    current_input=user_input,
                    prev_cmd_str=_read_logged_geometric_cmd(log_path, run_dir=base),
                    intended_cmd_str=intended_cmd_str,
                    label=f"TS in {base}",
                    qm_program=qm_program,
                )
                if decision == "invalidate":
                    _rename_conflicting_cache_dir(base)
                elif decision == "fail":
                    _raise_failed_cache(
                        f"TS optimization in {base}",
                        log_path,
                        error_cls=WorkflowError,
                    )
                elif decision == "reuse":
                    _log(f"Reusing cached TS result from {base}", level=1)
                    reused = True
                elif not _check_log_for_caching(log_path):
                    _log(f"{log_path.name} shows no convergence marker; will re-run TS optimization.", level=1)
        except Exception as cache_err:
            _log(f"TS cache check failed ({cache_err}); will re-optimize.", level=1)

    if not reused:
        base.mkdir(parents=True, exist_ok=True)
        shutil.copy(user_input, dest_input)

        single = GeoM()
        single.elem = list(M.elem) if hasattr(M, "elem") else []
        single.xyzs = [M.xyzs[frame_num].copy()]
        if hasattr(M, "comms") and M.comms and len(M.comms) > frame_num:
            single.comms = [M.comms[frame_num]]
        else:
            single.comms = [f"TS guess from {xyz_input.name}"]
        single.write(str(coords_xyz))

        cmd = _build_geometric_cmd(
            "ts",
            _geometric_argv_path(dest_input, base),
            _geometric_argv_path(coords_xyz, base),
            qm_program,
            ts_cmd_kwargs,
            nt=nt,
            prefix=user_prefix,
        )

        _log(f"Running: {cmd}", level=1)
        _log(f"Working directory: {base}", level=1)
        _write_last_command(base, cmd)

        result = subprocess.run(
            cmd,
            cwd=str(base),
            capture_output=True,
            text=True,
            **_subprocess_timeout_kwargs(ts_cmd_kwargs),
        )

        if result.returncode != 0:
            _log_warn("geometric TS optimization failed. stderr tail:")
            _log_warn(result.stderr[-2500:] if result.stderr else "(no stderr)")
            raise WorkflowError(
                f"TS optimization did not converge (geomeTRIC exit {result.returncode}). "
                f"Check {log_path}."
            )
        if not _check_log_for_caching(log_path):
            raise WorkflowError(
                f"TS optimization finished without a convergence marker in {log_path.name}."
            )

    ts_mol, energy = _load_ts_from_run_dir(base, GeoM, prefix)

    if energy is not None:
        _log(f"TS optimization complete. Energy = {energy:.8f} Ha", level=1)
    else:
        _log("TS optimization complete (energy not parsed)", level=1)

    return ts_mol, energy


_IRC_MULTIPLE_IMAGINARY_NEEDLE = "more than one imaginary vibrational mode"
_IRC_RECALC_HESSIAN_MSG = (
    "More than 1 imaginary mode detected from the optimized TS structure. "
    "Re-calculating Hessian for the IRC calculation."
)


def _irc_failed_multiple_imaginary_modes(
    stderr: str = "",
    stdout: str = "",
    *,
    log_path: Optional[Path] = None,
) -> bool:
    """Return True when geomeTRIC IRC failed due to multiple imaginary modes."""
    combined = f"{stderr or ''}\n{stdout or ''}".lower()
    if _IRC_MULTIPLE_IMAGINARY_NEEDLE in combined:
        return True
    if log_path is not None and log_path.is_file():
        return _IRC_MULTIPLE_IMAGINARY_NEEDLE in log_path.read_text().lower()
    return False


def _clear_irc_partial_artifacts(base: Path, prefix: str) -> None:
    """Remove partial IRC outputs so a retry can start cleanly."""
    import shutil

    for name in (_prefix_log(prefix), _prefix_irc_traj(prefix), "last_command.txt"):
        path = base / name
        if path.is_file():
            path.unlink()
    for tmp_dir in base.glob(f"{prefix}.tmp"):
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_irc(
    input_file: str,
    xyz_file: str,
    *,
    qm_program: str = "psi4",
    hessian: Optional[str] = None,
    direction: str = "both",
    nt: Optional[int] = None,
    run_dir: Optional[str] = None,
    postopt: bool = False,
    **irc_kwargs,
):
    """
    Run an IRC calculation from an optimized transition state using geomeTRIC.

    Recommended usage after ``optimize_ts`` (when using ``run_dir="ts_run"``):

        result = run_irc(
            "psi4.in",
            "ts_run/ts_optim.xyz",
            qm_program="psi4",
            run_dir="irc",
            nt=4,
        )
        traj = result["trj"]
        energies = result["energy"]
        endpoints = result["endpoints"]

    By default the Hessian is calculated at the TS geometry (no ``--hessian``
    flag). Pass ``hessian="/path/to/hessian.txt"`` to reuse a file Hessian
    (e.g. from a prior TS optimization in the full workflow).

    When a file Hessian is used and geomeTRIC reports multiple imaginary
    vibrational modes, IRC is retried once without ``--hessian`` so the
    Hessian is recalculated at the TS geometry.

    Parameters
    ----------
    input_file : str
        QM input template (geometry in the file is overridden via ``--coords``).
    xyz_file : str or Path
        Coordinate file for the TS (typically ``<run_dir>/<input_stem>_optim.xyz``).
    hessian : str or Path, optional
        Path to a NumPy-format Hessian file. When omitted, geomeTRIC calculates
        the Hessian at the TS geometry.
    direction : str
        ``forward``, ``backward``, or ``both`` (default).
    run_dir : str or Path, optional
        Working directory. When given, used as-is. Otherwise defaults to the
        process cwd.
    postopt : bool
        When ``True``, optimize the IRC endpoint structures with
        ``optimize_frames`` in ``<run_dir>/postopt/`` and return those in
        ``endpoints``. The returned ``trj`` is then the concatenation
        reactant-minimum → IRC → product-minimum, built from the full
        ``*_optim.xyz`` trajectories plus the IRC path.
    prefix : str, optional
        geomeTRIC ``--prefix`` for log/artifact basenames. When omitted, geomeTRIC
        uses the input-file stem (e.g. ``psi4`` from ``psi4.in``). Also used for
        endpoint optimizations when ``postopt=True``.
    postopt_rmsd_threshold : float, optional
        Aligned RMSD threshold (Å) for dropping duplicate junction frames when
        concatenating post-opt and IRC trajectories. Default: ``0.1``.
    recalc_hessian_on_failure : bool, optional
        When ``True`` (default), retry IRC without a file Hessian if the first
        attempt fails with multiple imaginary vibrational modes.

    Returns
    -------
    dict
        ``trj`` : IRC trajectory (``Molecule``), or opt + IRC + opt when
        ``postopt=True``
        ``energy`` : per-frame energies (``list[float | None]``)
        ``endpoints`` : two-frame ``Molecule`` (raw IRC endpoints, or optimized
        when ``postopt=True``)

    When ``postopt=True``, also writes ``<prefix>_irc_postopt.xyz`` in ``run_dir``
    (e.g. ``irc_irc_postopt.xyz``) with the concatenated opt + IRC + opt path.
    """
    verbose = _pop_verbose(irc_kwargs)
    irc_cmd_kwargs = {**irc_kwargs, "verbose": verbose}
    _log(f"Running IRC (direction={direction})", level=1)

    import shutil
    import subprocess
    from pathlib import Path

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for run_irc in direct mode.")

    qm_program = irc_kwargs.pop("qm_program", None) or qm_program or "psi4"
    inp_file = irc_kwargs.pop("input_file", None) or input_file
    if inp_file is None:
        raise ValueError("run_irc requires `input_file` (QM template).")

    user_input = Path(inp_file)
    if not user_input.exists():
        raise FileNotFoundError(f"Input file not found: {inp_file}")

    direction = irc_kwargs.pop("direction", direction) or "both"
    hessian = irc_kwargs.pop("hessian", None) or hessian
    recalc_hessian_on_failure = irc_kwargs.pop("recalc_hessian_on_failure", True)
    if not xyz_file:
        raise ValueError("run_irc requires `xyz_file` (e.g. ts_run/ts_optim.xyz).")

    xyz_p = Path(xyz_file)
    if not xyz_p.exists():
        raise FileNotFoundError(f"xyz_file not found: {xyz_file}")
    M = GeoM(str(xyz_p))
    src_xyz = M.xyzs[-1] if len(M.xyzs) > 1 else M.xyzs[0]
    hessian_path: Optional[Path]
    if hessian is None:
        hessian_path = None
    else:
        hessian_path = Path(hessian)
        if not hessian_path.is_file():
            raise FileNotFoundError(f"Hessian file not found: {hessian_path}")

    user_prefix = irc_kwargs.pop("prefix", None)
    base = _resolve_run_dir(run_dir)
    dest_input = base / user_input.name
    prefix = _output_prefix(dest_input, user_prefix)
    log_path = base / _prefix_log(prefix)
    traj_path = base / _prefix_irc_traj(prefix)
    coords_for_cmd = base / xyz_p.name

    hessian_for_cmd = hessian_path.resolve() if hessian_path is not None else None

    intended_cmd_list = _build_geometric_cmd(
        "irc",
        _geometric_argv_path(dest_input, base),
        _geometric_argv_path(coords_for_cmd, base),
        qm_program,
        irc_cmd_kwargs,
        nt=nt,
        prefix=user_prefix,
        hessian_path=hessian_for_cmd,
        irc_direction=direction,
    )
    intended_cmd_str = " ".join(intended_cmd_list)

    reused = False
    if (base.exists()
            and dest_input.exists()
            and coords_for_cmd.exists()
            and traj_path.exists()
            and log_path.exists()):
        try:
            cached_m = GeoM(str(coords_for_cmd))
            if cached_m.xyzs and np.allclose(
                cached_m.xyzs[-1] if len(cached_m.xyzs) > 1 else cached_m.xyzs[0],
                src_xyz,
                atol=1e-8,
            ):
                decision = _converged_cache_decision(
                    log_path=log_path,
                    cached_input=dest_input,
                    current_input=user_input,
                    prev_cmd_str=_read_logged_geometric_cmd(log_path, run_dir=base),
                    intended_cmd_str=intended_cmd_str,
                    label=f"IRC in {base}",
                    qm_program=qm_program,
                )
                if decision == "invalidate":
                    _rename_conflicting_cache_dir(base)
                elif decision == "fail":
                    if (
                        recalc_hessian_on_failure
                        and hessian_for_cmd is not None
                        and _irc_failed_multiple_imaginary_modes(log_path=log_path)
                    ):
                        _log(_IRC_RECALC_HESSIAN_MSG, level=0)
                        _clear_irc_partial_artifacts(base, prefix)
                    else:
                        _raise_failed_cache(f"IRC in {base}", log_path)
                elif decision == "reuse":
                    cached_traj = GeoM(str(traj_path))
                    if len(getattr(cached_traj, "xyzs", [])) >= 2:
                        _log(f"Reusing cached IRC result from {base}", level=1)
                        reused = True
                    else:
                        _log(f"{traj_path.name} has fewer than 2 frames; will re-run IRC.", level=1)
                elif not _check_log_for_caching(log_path):
                    _log(f"{log_path.name} shows no convergence marker; will re-run IRC.", level=1)
        except Exception as cache_err:
            _log(f"IRC cache check failed ({cache_err}); will re-run.", level=1)

    if not reused:
        base.mkdir(parents=True, exist_ok=True)
        shutil.copy(user_input, dest_input)

        shutil.copy(xyz_p, coords_for_cmd)

        use_hessian_path: Optional[Path] = hessian_for_cmd
        while True:
            cmd = _build_geometric_cmd(
                "irc",
                _geometric_argv_path(dest_input, base),
                _geometric_argv_path(coords_for_cmd, base),
                qm_program,
                irc_cmd_kwargs,
                nt=nt,
                prefix=user_prefix,
                hessian_path=use_hessian_path,
                irc_direction=direction,
            )

            _log(f"Running: {cmd}", level=1)
            _log(f"Working directory: {base}", level=1)
            _write_last_command(base, cmd)

            result = subprocess.run(
                cmd,
                cwd=str(base),
                capture_output=True,
                text=True,
                **_subprocess_timeout_kwargs(irc_cmd_kwargs),
            )

            if result.returncode == 0:
                break

            if (
                use_hessian_path is not None
                and recalc_hessian_on_failure
                and _irc_failed_multiple_imaginary_modes(
                    result.stderr or "",
                    result.stdout or "",
                    log_path=log_path,
                )
            ):
                _log(_IRC_RECALC_HESSIAN_MSG, level=0)
                _clear_irc_partial_artifacts(base, prefix)
                use_hessian_path = None
                continue

            _log_warn("geometric IRC failed. stderr tail:")
            _log_warn(result.stderr[-2500:] if result.stderr else "(no stderr)")
            if traj_path.exists():
                try:
                    traj, energies = _load_irc_from_run_dir(base, GeoM, prefix)
                    return _package_irc_result(
                        traj,
                        energies,
                        GeoM,
                        postopt=postopt,
                        base=base,
                        input_file=str(inp_file),
                        qm_program=qm_program,
                        nt=nt,
                        prefix=user_prefix,
                        irc_kwargs=irc_cmd_kwargs,
                    )
                except Exception:
                    pass

            raise RuntimeError(
                f"IRC calculation failed. Check {log_path}."
            )

    traj, energies = _load_irc_from_run_dir(base, GeoM, prefix)

    re_e = energies[0] if energies else None
    pr_e = energies[-1] if energies else None
    if re_e is not None or pr_e is not None:
        msg = f"IRC complete ({len(traj.xyzs)} frames"
        if re_e is not None:
            msg += f"; reactant = {re_e:.8f} Ha"
        if pr_e is not None:
            msg += f"; product = {pr_e:.8f} Ha"
        msg += ")"
        _log(msg, level=0)
    else:
        _log(f"IRC complete ({len(traj.xyzs)} frames; energies not parsed)", level=0)

    return _package_irc_result(
        traj,
        energies,
        GeoM,
        postopt=postopt,
        base=base,
        input_file=str(inp_file),
        qm_program=qm_program,
        nt=nt,
        prefix=user_prefix,
        irc_kwargs=irc_cmd_kwargs,
    )


def _count_imaginary_from_vdata(vdata_path: Path) -> Optional[int]:
    """Return the number of imaginary vibrational modes parsed from a vdata file."""
    if not vdata_path.is_file():
        return None
    match = re.search(r"(\d+)\s+Imaginary Frequencies", vdata_path.read_text())
    if match:
        return int(match.group(1))
    return 0


def _count_imaginary_modes_from_hessian(
    coords_ang: np.ndarray,
    elem: Sequence[str],
    hessian_path: Union[str, Path],
) -> int:
    """Count imaginary modes from a Cartesian Hessian and optimized geometry."""
    import tempfile

    from geometric.nifty import ang2bohr
    from geometric.normal_modes import frequency_analysis

    hessian = np.loadtxt(hessian_path)
    coords = np.asarray(coords_ang, dtype=float).reshape(-1, 3) * ang2bohr
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        frequency_analysis(coords, hessian, elem=list(elem), verbose=0, outfnm=str(out_path))
        return _count_imaginary_from_vdata(out_path) or 0
    finally:
        out_path.unlink(missing_ok=True)


def _count_imaginary_modes_from_ts_run(
    run_dir: Path,
    prefix: str,
    GeoM,
) -> int:
    """Return the number of imaginary modes for a completed TS optimization."""
    vdata_path = run_dir / f"{prefix}.vdata_last"
    count = _count_imaginary_from_vdata(vdata_path)
    if count is not None:
        return count

    hessian_path = run_dir / f"{prefix}.tmp" / "hessian" / "hessian.txt"
    opt_path = run_dir / _prefix_optim_xyz(prefix)
    if not (hessian_path.is_file() and opt_path.is_file()):
        raise WorkflowError(
            f"Cannot count imaginary modes in {run_dir}: "
            f"missing {prefix}.vdata_last or Hessian/optim artifacts."
        )
    mol = GeoM(str(opt_path))
    return _count_imaginary_modes_from_hessian(mol.xyzs[-1], mol.elem, hessian_path)


def run_neb_with_ts_opt(
    input_file: str,
    initial_chain: Union[str, Path],
    *,
    qm_program: str = "psi4",
    nt: Optional[int] = None,
    run_dir: Optional[Union[str, Path]] = None,
    ts_run_dir: Optional[Union[str, Path]] = None,
    neb_kwargs: Optional[Dict[str, Any]] = None,
    ts_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run NEB and optimize the climbing-image TS guess when one is found.

    Returns the NEB result plus optional TS optimization outputs. When a TS is
    optimized, ``ts_xyz`` points to ``<prefix>_optim.xyz`` under ``ts_run_dir``.
    """
    neb_kw = dict(neb_kwargs or {})
    ts_kw = dict(ts_kwargs or {})

    neb_dir = _resolve_run_dir(run_dir)
    neb_prefix = str(neb_kw.get("prefix", "neb"))
    ts_prefix = str(ts_kw.get("prefix", "ts"))

    neb_result = run_neb(
        input_file,
        initial_chain,
        qm_program=qm_program,
        nt=nt,
        run_dir=str(neb_dir),
        **neb_kw,
    )

    ts_mol = None
    ts_energy = None
    ts_xyz: Optional[Path] = None

    if neb_result.get("ts_guess") is not None:
        ts_climb = neb_dir / _prefix_neb_ts_climb(neb_prefix)
        if ts_run_dir is None:
            ts_dir = neb_dir.parent / "ts_run"
        else:
            ts_dir = Path(ts_run_dir)
        ts_dir.mkdir(parents=True, exist_ok=True)

        ts_mol, ts_energy = optimize_ts(
            input_file,
            str(ts_climb),
            qm_program=qm_program,
            nt=nt,
            run_dir=str(ts_dir),
            **ts_kw,
        )
        ts_xyz = ts_dir / _prefix_optim_xyz(ts_prefix)
        _log(f"Optimized TS written → {ts_xyz}", level=0)
    else:
        _log("NEB found no climbing image; TS optimization skipped.", level=0)

    return {
        "neb_result": neb_result,
        "ts_mol": ts_mol,
        "ts_energy": ts_energy,
        "ts_xyz": ts_xyz,
    }


def run_irc_postopt(
    input_file: str,
    ts_xyz: Union[str, Path],
    *,
    qm_program: str = "psi4",
    hessian: Optional[Union[str, Path]] = None,
    nt: Optional[int] = None,
    run_dir: Optional[Union[str, Path]] = None,
    **irc_kwargs,
) -> Dict[str, Any]:
    """
    Run IRC from an optimized TS and post-optimize the IRC endpoints.

    Writes ``<prefix>_irc_postopt.xyz`` in ``run_dir`` (opt trajectories +
    IRC path concatenated).
    """
    irc_kw = dict(irc_kwargs)
    irc_kw.setdefault("postopt_rmsd_threshold", 0.1)
    base = _resolve_run_dir(run_dir)
    prefix = irc_kw.get("prefix", "irc")

    result = run_irc(
        input_file,
        str(ts_xyz),
        qm_program=qm_program,
        hessian=str(hessian) if hessian is not None else None,
        nt=nt,
        run_dir=str(base),
        postopt=True,
        **irc_kw,
    )
    artifact_prefix = _output_prefix(base / Path(input_file).name, prefix)
    postopt_path = base / _prefix_irc_postopt_traj(artifact_prefix)
    result["postopt_xyz"] = postopt_path
    return result


def optimize_ts_with_postirc(
    input_file: str,
    xyz_file: Union[str, Path],
    *,
    postirc: bool = False,
    qm_program: str = "psi4",
    nt: Optional[int] = None,
    run_dir: Optional[Union[str, Path]] = None,
    irc_run_dir: Optional[Union[str, Path]] = None,
    ts_kwargs: Optional[Dict[str, Any]] = None,
    irc_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Optimize a transition-state guess with ``--hessian first+last``.

    When ``postirc=True`` and the optimized structure has exactly one imaginary
    mode, also run :func:`run_irc_postopt` and return the post-opt IRC pathway.
    """
    ts_kw = dict(ts_kwargs or {})
    irc_kw = dict(irc_kwargs or {})
    base = _resolve_run_dir(run_dir)
    ts_prefix = str(ts_kw.get("prefix", "ts"))

    ts_mol, ts_energy = optimize_ts(
        input_file,
        str(xyz_file),
        qm_program=qm_program,
        nt=nt,
        run_dir=str(base),
        **ts_kw,
    )
    ts_xyz = base / _prefix_optim_xyz(ts_prefix)

    result: Dict[str, Any] = {
        "ts_mol": ts_mol,
        "ts_energy": ts_energy,
        "ts_xyz": ts_xyz,
        "n_imaginary_modes": None,
        "irc_result": None,
        "postopt_xyz": None,
    }

    if not postirc:
        return result

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for post-IRC analysis.")

    n_imag = _count_imaginary_modes_from_ts_run(base, ts_prefix, GeoM)
    result["n_imaginary_modes"] = n_imag
    if n_imag != 1:
        _log_warn(
            f"Skipping IRC: expected exactly 1 imaginary mode, found {n_imag}."
        )
        return result

    if irc_run_dir is None:
        irc_dir = base.parent / "irc_run"
    else:
        irc_dir = Path(irc_run_dir)

    ts_hessian = base / f"{ts_prefix}.tmp" / "hessian" / "hessian.txt"
    irc_result = run_irc_postopt(
        input_file,
        ts_xyz,
        qm_program=qm_program,
        hessian=ts_hessian,
        nt=nt,
        run_dir=str(irc_dir),
        **irc_kw,
    )
    result["irc_result"] = irc_result
    result["postopt_xyz"] = irc_result.get("postopt_xyz")
    return result


def _drms_aligned(a, b) -> float:
    """Aligned RMSD (Å) between two coordinate arrays."""
    from geometric.step import calc_drms_dmax
    from geometric.nifty import ang2bohr

    return float(
        calc_drms_dmax(np.asarray(a) * ang2bohr, np.asarray(b) * ang2bohr, align=True)[0]
    )


def _build_tric_primitives(GeoM, elem, xyz):
    """Build TRIC primitive internal coordinates on a single-frame geometry."""
    from geometric.internal import PrimitiveInternalCoordinates

    mol = GeoM()
    mol.elem = list(elem)
    mol.xyzs = [np.asarray(xyz, dtype=float).reshape(-1, 3).copy()]
    return PrimitiveInternalCoordinates(
        mol, build=True, connect=False, addcart=False, connect_isolated=False,
    )


def _topology_unique_primitives(IC_src, IC_ref) -> list:
    """Primitives present in *IC_src* but not in *IC_ref* (type + atom indices only)."""
    return [p for p in IC_src.Internals if p not in IC_ref.Internals]


def _topology_overlap_count(primitives, IC_target) -> int:
    """Count how many primitive topologies from *primitives* exist in *IC_target*."""
    return sum(1 for p in primitives if p in IC_target.Internals)


def _topology_pairing_score(unique_ep, IC_target_ep, IC_full_ep) -> int:
    """Score one endpoint assignment using unique-then-full primitive topology overlap."""
    if unique_ep:
        return _topology_overlap_count(unique_ep, IC_target_ep)
    return _topology_overlap_count(IC_full_ep.Internals, IC_target_ep)


def _angle_pairing_cost(
    xyz_ep: np.ndarray,
    IC_ep,
    xyz_target: np.ndarray,
    IC_target,
) -> float:
    """Sum of |angle_ep - angle_target| over shared ``Angle`` primitives (lower is better)."""
    from geometric.internal import Angle

    xyz_ep = np.asarray(xyz_ep, dtype=float).reshape(-1, 3)
    xyz_target = np.asarray(xyz_target, dtype=float).reshape(-1, 3)
    total = 0.0
    n_angles = 0
    for prim in IC_ep.Internals:
        if type(prim) is not Angle:
            continue
        if prim not in IC_target.Internals:
            continue
        total += abs(float(prim.value(xyz_ep)) - float(prim.value(xyz_target)))
        n_angles += 1
    if n_angles == 0:
        return float("inf")
    return total


def _pic_orient_pairing_by_angles(
    ep0: np.ndarray,
    ep1: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    IC_ep0,
    IC_ep1,
    IC_a,
    IC_b,
) -> bool:
    """
    Orient by ``Angle`` primitive values when IRC endpoints share identical PIC topology.

    Returns ``True`` when flipped assignment (ep0→b, ep1→a) has lower total angle
    deviation than direct (ep0→a, ep1→b). Ties default to direct.
    """
    cost_direct = (
        _angle_pairing_cost(ep0, IC_ep0, target_a, IC_a)
        + _angle_pairing_cost(ep1, IC_ep1, target_b, IC_b)
    )
    cost_flipped = (
        _angle_pairing_cost(ep0, IC_ep0, target_b, IC_b)
        + _angle_pairing_cost(ep1, IC_ep1, target_a, IC_a)
    )
    _log(
        "PIC identical topology; angle costs (lower=better): "
        f"ep0→a/ep1→b={cost_direct:.6f}, ep0→b/ep1→a={cost_flipped:.6f}",
        level=1,
    )
    if cost_flipped < cost_direct:
        return True
    return False


def _pic_orient_pairing(ep0, ep1, target_a, target_b, elem, GeoM) -> bool:
    """
    Decide whether IRC ep0/ep1 align with optimized targets (a, b) or flipped (b, a).

    Compares TRIC primitive *topology* only (``Internals`` membership), not values.
    Endpoint-unique primitives are those in one IRC-end IC but not the other;
    each unique set is matched against the optimized endpoint ICs. When both IRC
    endpoints share the same PIC topology, ``Angle`` primitive *values* are
    compared against the targets instead.

    Returns
    -------
    bool
        ``False`` for ep0→a / ep1→b, ``True`` for ep0→b / ep1→a. Ties default to
        the direct assignment.
    """
    IC_ep0 = _build_tric_primitives(GeoM, elem, ep0)
    IC_ep1 = _build_tric_primitives(GeoM, elem, ep1)
    IC_a = _build_tric_primitives(GeoM, elem, target_a)
    IC_b = _build_tric_primitives(GeoM, elem, target_b)

    unique_ep0 = _topology_unique_primitives(IC_ep0, IC_ep1)
    unique_ep1 = _topology_unique_primitives(IC_ep1, IC_ep0)

    if not unique_ep0 and not unique_ep1:
        return _pic_orient_pairing_by_angles(
            ep0, ep1, target_a, target_b, IC_ep0, IC_ep1, IC_a, IC_b,
        )

    score_direct = (
        _topology_pairing_score(unique_ep0, IC_a, IC_ep0)
        + _topology_pairing_score(unique_ep1, IC_b, IC_ep1)
    )
    score_flipped = (
        _topology_pairing_score(unique_ep0, IC_b, IC_ep0)
        + _topology_pairing_score(unique_ep1, IC_a, IC_ep1)
    )
    _log(
        "PIC topology scores: "
        f"ep0→a/ep1→b={score_direct} ({len(unique_ep0)}+{len(unique_ep1)} unique prims), "
        f"ep0→b/ep1→a={score_flipped}",
        level=1,
    )
    if score_flipped > score_direct:
        return True
    return False


def _frame_xyz(geom) -> np.ndarray:
    if hasattr(geom, "xyzs"):
        return geom.xyzs[0].copy()
    return np.asarray(geom).copy()


_DEFAULT_NEB_KWARGS: Dict[str, Any] = {
    "n_images": 11,
    "maxg": 0.1,
    "avgg": 0.05,
    "coordsys": "tric",
    "prefix": "neb",
}
_DEFAULT_OPT_KWARGS: Dict[str, Any] = {
    "coordsys": "tric",
    "prefix": "optimize",
}
_DEFAULT_TS_KWARGS: Dict[str, Any] = {
    "coordsys": "tric",
    "prefix": "ts",
}
_DEFAULT_IRC_KWARGS: Dict[str, Any] = {
    "coordsys": "tric",
    "prefix": "irc",
}
_DEFAULT_INTERP_KWARGS: Dict[str, Any] = {
    "n_images": 50,
    "align_system": True,
    "align_frags": False,
    "fast": False,
}


def _merge_calc_kwargs(
    user: Optional[Dict[str, Any]],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of *defaults* updated with user-supplied calculation kwargs."""
    out = dict(defaults)
    if user:
        out.update(user)
    return out


class TRICWorkflow:
    """
    Discover a refined reaction pathway between two input minima.

    Optimizes the endpoints, then repeatedly runs TRICS interpolate → NEB → TS opt →
    IRC (with endpoint re-optimization) until IRC endpoints match the dynamic
    pathway targets (aligned RMSD < ``rmsd_threshold``). Targets start as the
    optimized input endpoints; when one IRC end matches a target label, that
    label moves to the other IRC end while the unmatched label stays at the
    original global minimum until it is matched. When NEB finds no climbing
    image (no energy barrier), the converged NEB chain is used directly as that
    elementary segment instead of running TS optimization and IRC. When only one
    IRC/NEB end matches, a further elementary step is launched toward the unmatched
    minimum. When neither matches, the segment is oriented by matching TRIC
    primitive topology (``Internals`` membership, not values) at the IRC endpoints
    to the optimized endpoint frames, then both directions are refined recursively.

    Writes ``full_pathway.xyz`` under ``work_dir`` and returns the merged pathway
    as a geomeTRIC ``Molecule``. Barrier-crossing elementary segments include the
    full post-optimization trajectories at both endpoints (opt + IRC + opt).

    Pass geomeTRIC options per calculation type via ``neb``, ``opt``, ``ts``,
    ``irc``, and ``interp`` dicts. Each dict is forwarded as ``**kwargs`` to the
    corresponding geomeTRIC driver (e.g. ``opt={"converge": "set GAU"}``,
    ``interp={"align_frags": True}``). Keys such as ``converge``, ``coordsys``,
    ``maxiter``, ``trust``, and ``prefix`` follow geomeTRIC naming.

    IRC endpoint post-optimization reuses ``opt["converge"]`` unless
    ``irc["postopt_converge"]`` is set explicitly.

    Individual calculation stages are exposed as module-level functions:
    :func:`optimize_frames`, :func:`interpolate`, :func:`run_neb`,
    :func:`optimize_ts`, :func:`run_irc`, and :func:`get_energies`.

    Set ``verbose=0`` (default) for step summaries only; ``verbose=1`` also prints
    commands, working directories, cache reuse, and forwards ``--verbose`` to
    geomeTRIC when not overridden in a calculation dict.
    """

    def __init__(
        self,
        input_file: Union[str, Path],
        *,
        qm_program: str = "psi4",
        work_dir: Optional[Union[str, Path]] = None,
        nt: Optional[int] = None,
        rmsd_threshold: float = 0.1,
        max_depth: int = 10,
        verbose: int = 0,
        neb: Optional[Dict[str, Any]] = None,
        opt: Optional[Dict[str, Any]] = None,
        ts: Optional[Dict[str, Any]] = None,
        irc: Optional[Dict[str, Any]] = None,
        interp: Optional[Dict[str, Any]] = None,
    ):
        self.input_file = Path(input_file).resolve()
        self.qm_program = qm_program
        self.work_dir = Path(work_dir).resolve() if work_dir is not None else Path.cwd().resolve()
        self.nt = nt
        self.rmsd_threshold = rmsd_threshold
        self.max_depth = max_depth
        self.verbose = int(verbose)
        self.neb = _merge_calc_kwargs(neb, _DEFAULT_NEB_KWARGS)
        self.opt = _merge_calc_kwargs(opt, _DEFAULT_OPT_KWARGS)
        self.ts = _merge_calc_kwargs(ts, _DEFAULT_TS_KWARGS)
        self.irc = _merge_calc_kwargs(irc, _DEFAULT_IRC_KWARGS)
        self.interp = _merge_calc_kwargs(interp, _DEFAULT_INTERP_KWARGS)

        self._step_counter = 0
        self._discovered_ts: List[np.ndarray] = []
        self._elem: List[str] = []

        GeoM = _get_geo_molecule()
        if GeoM is None:
            raise RuntimeError("geomeTRIC is required for TRICWorkflow.")
        self._GeoM = GeoM

        if not self.input_file.is_file():
            raise FileNotFoundError(f"QM input file not found: {self.input_file}")

    @property
    def opt_prefix(self) -> str:
        return str(self.opt.get("prefix", "optimize"))

    @property
    def neb_prefix(self) -> str:
        return str(self.neb.get("prefix", "neb"))

    @property
    def ts_prefix(self) -> str:
        return str(self.ts.get("prefix", "ts"))

    @property
    def irc_prefix(self) -> str:
        return str(self.irc.get("prefix", "irc"))

    @property
    def interp_n_images(self) -> int:
        return int(self.interp.get("n_images", 50))

    def _calc_kwargs(self, role: str, **overrides) -> Dict[str, Any]:
        """Return a copy of the workflow calculation kwargs for *role*."""
        store = {
            "neb": self.neb,
            "opt": self.opt,
            "ts": self.ts,
            "irc": self.irc,
            "interp": self.interp,
        }[role]
        out = dict(store)
        out.update(overrides)
        out.setdefault("verbose", self.verbose)
        return out

    def _opt_cmd_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments used for geomeTRIC optimize command/cache comparison."""
        kw = self._calc_kwargs("opt")
        kw.pop("prefix", None)
        return kw

    def _irc_run_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments forwarded to :func:`run_irc` from the workflow."""
        irc_kwargs = self._calc_kwargs("irc")
        if "converge" in self.opt and "postopt_converge" not in irc_kwargs:
            irc_kwargs["postopt_converge"] = self.opt["converge"]
        trust = irc_kwargs.get("trust")
        if trust is not None:
            _log(f"IRC trust radius = {trust}", level=1)
        return irc_kwargs

    def _optimize_endpoints(self, xyz_file: Union[str, Path]) -> "GeoMolecule":
        """Optimize frames in *xyz_file* and write ``optimized_endpoints.xyz``."""
        xyz_path = Path(xyz_file)
        if not xyz_path.is_file():
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
        if len(self._GeoM(str(xyz_path))) < 2:
            raise WorkflowError("Input XYZ must contain at least two frames.")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        opt_path = self.work_dir / "optimized_endpoints.xyz"

        with _log_context(verbosity=self.verbose):
            opt_mols = self._try_load_cached_endpoints(xyz_path)
            if opt_mols is not None:
                _log(f"Reusing cached optimized endpoints → {opt_path.name}", level=0)
            else:
                _log(f"Optimizing endpoints from {xyz_path.name}...", level=0)
                opt_dir = self.work_dir / "opt_runs"
                opt_mols = optimize_frames(
                    str(self.input_file),
                    str(xyz_path),
                    qm_program=self.qm_program,
                    run_dir=str(opt_dir),
                    nt=self.nt,
                    **self._calc_kwargs("opt"),
                )
                opt_mols.write(str(opt_path))
                _log(f"Endpoint optimization complete → {opt_path.name}", level=0)
        return opt_mols

    def run(self, xyz_file: Union[str, Path]) -> "GeoMolecule":
        """Optimize two frames in *xyz_file* and discover the connecting pathway."""
        xyz_path = Path(xyz_file)
        if not xyz_path.is_file():
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter = 0
        self._discovered_ts.clear()

        with _log_context(verbosity=self.verbose):
            opt_mols = self._optimize_endpoints(xyz_path)
            self._elem = list(getattr(opt_mols, "elem", []) or [])

            initial_target_a = opt_mols.xyzs[0].copy()
            initial_target_b = opt_mols.xyzs[-1].copy()
            start_mol = opt_mols[0]
            end_mol = opt_mols[-1]

            _log("Discovering elementary reaction pathway...", level=0)
            pathway = self._solve(
                start_mol,
                end_mol,
                initial_target_a.copy(),
                initial_target_b.copy(),
                depth=0,
            )
            pathway = self._anchor_pathway_endpoints(
                pathway, initial_target_a, initial_target_b,
            )
            self._verify_pathway_connectivity(
                pathway, initial_target_a, initial_target_b,
            )

            out_path = self.work_dir / "full_pathway.xyz"

            pathway.align()
            pathway.write(str(out_path))
            _log(
                f"Pathway complete ({len(pathway.xyzs)} frames) → {out_path.name}",
                level=0,
            )
            return pathway

    def _step_artifact_paths(self, step_id: int):
        """
        Per-step artifact layout under work_dir.

        step_id N → step_{N:02d}/endpoints.xyz, interpolated.xyz,
                    neb_run/, ts_run/, irc_run/
        """
        step_dir = self.work_dir / f"step_{step_id:02d}"
        return {
            "label": f"step_{step_id:02d}",
            "endpoints_xyz": step_dir / "endpoints.xyz",
            "interp_xyz": step_dir / "interpolated.xyz",
            "neb_dir": step_dir / "neb_run",
            "ts_dir": step_dir / "ts_run",
            "irc_dir": step_dir / "irc_run",
        }

    def _elem_for_batch(self, *mols) -> List[str]:
        """Return element symbols from the first molecule that defines them."""
        for mol in mols:
            elem = list(getattr(mol, "elem", []) or [])
            if elem:
                return elem
        return list(self._elem)

    def _single_frame_mol(self, geom, comment: str = "frame"):
        mol = self._GeoM()
        if hasattr(geom, "elem") and getattr(geom, "elem", None):
            mol.elem = list(geom.elem)
        else:
            mol.elem = list(self._elem)
        mol.xyzs = [_frame_xyz(geom)]
        mol.comms = [comment]
        return mol

    def _write_two_frame_xyz(self, path, mol_a, mol_b, label_a="start", label_b="end"):
        batch = self._GeoM()
        batch.elem = self._elem_for_batch(mol_a, mol_b)
        batch.xyzs = [_frame_xyz(mol_a), _frame_xyz(mol_b)]
        batch.comms = [label_a, label_b]
        batch.write(str(path))
        return path

    def _write_two_frame_xyz_if_changed(self, path, mol_a, mol_b, label_a="start", label_b="end"):
        """Write endpoints only when coordinates differ (preserves cache mtimes)."""
        start_xyz = _frame_xyz(mol_a)
        end_xyz = _frame_xyz(mol_b)
        if _two_frame_xyz_matches(path, start_xyz, end_xyz, self._GeoM):
            return path
        return self._write_two_frame_xyz(path, mol_a, mol_b, label_a, label_b)

    def _try_load_cached_endpoints_from_artifacts(self, initial_xyz: Path):
        """Reuse ``optimized_endpoints.xyz`` when per-frame opt artifacts still match."""
        opt_path = self.work_dir / "optimized_endpoints.xyz"
        opt_dir = self.work_dir / "opt_runs"
        if not opt_path.is_file():
            return None

        initial = self._GeoM(str(initial_xyz))
        n_frames = len(getattr(initial, "xyzs", []) or [])
        if n_frames < 2:
            return None

        for frame_num in range(n_frames):
            frame_dir = opt_dir / f"frame_{frame_num}"
            dest_input = frame_dir / self.input_file.name
            prefix = _output_prefix(dest_input, self.opt_prefix)
            if not _optimization_artifact_cache_hit(
                frame_dir,
                self.input_file,
                initial.xyzs[frame_num],
                prefix,
                self._GeoM,
                qm_program=self.qm_program,
            ):
                return None

        try:
            cached = self._GeoM(str(opt_path))
        except Exception:
            return None
        if len(getattr(cached, "xyzs", []) or []) < 2:
            return None
        return cached

    def _try_load_cached_endpoints(self, initial_xyz: Path):
        """Return optimized endpoint Molecule when opt_runs cache is still valid."""
        opt_path = self.work_dir / "optimized_endpoints.xyz"
        opt_dir = self.work_dir / "opt_runs"
        if not opt_path.is_file():
            return None

        initial = self._GeoM(str(initial_xyz))
        n_frames = len(getattr(initial, "xyzs", []) or [])
        if n_frames < 2:
            return None

        strict_ok = True
        for frame_num in range(n_frames):
            frame_dir = opt_dir / f"frame_{frame_num}"
            coords_xyz = frame_dir / "coords.xyz"
            dest_input = frame_dir / self.input_file.name
            prefix = _output_prefix(dest_input, self.opt_prefix)
            log_path = frame_dir / _prefix_log(prefix)
            opt_xyz = frame_dir / _prefix_optim_xyz(prefix)

            if not (coords_xyz.is_file() and dest_input.is_file() and opt_xyz.is_file()):
                return None
            if not _qm_chemistry_matches(
                dest_input,
                self.input_file,
                qm_program=self.qm_program,
            ):
                return None
            try:
                cached_coords = self._GeoM(str(coords_xyz))
                if not cached_coords.xyzs or not np.allclose(
                    cached_coords.xyzs[0], initial.xyzs[frame_num], atol=1e-8,
                ):
                    return None
            except Exception:
                return None
            if _find_converged_log_in_dir(frame_dir, preferred=log_path) is None:
                if not _optimization_artifact_cache_hit(
                    frame_dir,
                    self.input_file,
                    initial.xyzs[frame_num],
                    prefix,
                    self._GeoM,
                    qm_program=self.qm_program,
                ):
                    strict_ok = False
                    break

        if strict_ok:
            try:
                cached = self._GeoM(str(opt_path))
            except Exception:
                return None
            if len(getattr(cached, "xyzs", []) or []) < 2:
                return None
            return cached

        return self._try_load_cached_endpoints_from_artifacts(initial_xyz)

    def _reverse_trajectory(self, mol):
        return _reverse_trajectory(mol, self._GeoM)

    def _orient_trajectory(self, mol, start_xyz, end_xyz):
        """Return *mol* oriented so frame 0 → *start_xyz* and frame -1 → *end_xyz*."""
        forward_cost = (
            _drms_aligned(mol.xyzs[0], start_xyz)
            + _drms_aligned(mol.xyzs[-1], end_xyz)
        )
        reverse_cost = (
            _drms_aligned(mol.xyzs[-1], start_xyz)
            + _drms_aligned(mol.xyzs[0], end_xyz)
        )
        if reverse_cost < forward_cost:
            mol = self._reverse_trajectory(mol)
        return mol

    def _resolve_target_pairing(self, ep0, ep1, target_a, target_b, elem):
        """
        Choose whether IRC ep0/ep1 correspond to optimized targets (a, b) or (b, a).

        Uses TRIC primitive topology overlap only (no coordinate values or RMSD).
        """
        return _pic_orient_pairing(ep0, ep1, target_a, target_b, elem, self._GeoM)

    def _orient_trajectory_to_targets(self, trj, ep0, ep1, target_a, target_b):
        """Orient *trj* using :meth:`_resolve_target_pairing`."""
        flipped = self._resolve_target_pairing(
            ep0, ep1, target_a, target_b, list(trj.elem),
        )
        if flipped:
            return self._orient_trajectory(trj, target_b, target_a), flipped
        return self._orient_trajectory(trj, target_a, target_b), flipped

    def _concat_trajectories(self, traj_a, traj_b, junction_xyz):
        return _concat_trajectories(
            traj_a, traj_b, junction_xyz, self.rmsd_threshold, self._GeoM,
        )

    def _concat_segments(self, segments: List) -> "GeoMolecule":
        if not segments:
            raise WorkflowError("No IRC segments to merge.")
        out = segments[0]
        for seg in segments[1:]:
            if out.xyzs and seg.xyzs:
                seg.xyzs[0] = out.xyzs[-1].copy()
            out = self._concat_trajectories(out, seg, out.xyzs[-1])
        return out

    def _anchor_pathway_endpoints(self, pathway, target_a, target_b):
        """Force the merged pathway ends to match the optimized endpoint frames."""
        if getattr(pathway, "xyzs", None):
            pathway.xyzs[0] = np.asarray(target_a, dtype=float).reshape(-1, 3).copy()
            pathway.xyzs[-1] = np.asarray(target_b, dtype=float).reshape(-1, 3).copy()
        return pathway

    def _verify_pathway_connectivity(
        self,
        pathway,
        target_a,
        target_b,
        *,
        label: str = "pathway",
    ) -> None:
        """Raise when anchored endpoints or their first/last interior junctions are discontinuous."""
        xyzs = getattr(pathway, "xyzs", None) or []
        if len(xyzs) < 2:
            raise WorkflowError(f"{label}: pathway must contain at least two frames.")

        start_anchor = _drms_aligned(xyzs[0], target_a)
        end_anchor = _drms_aligned(xyzs[-1], target_b)
        if start_anchor >= self.rmsd_threshold:
            raise WorkflowError(
                f"{label}: pathway start does not match optimized endpoint A "
                f"(aligned RMSD = {start_anchor:.4f} Å >= {self.rmsd_threshold} Å)."
            )
        if end_anchor >= self.rmsd_threshold:
            raise WorkflowError(
                f"{label}: pathway end does not match optimized endpoint B "
                f"(aligned RMSD = {end_anchor:.4f} Å >= {self.rmsd_threshold} Å)."
            )

        start_jump = _drms_aligned(xyzs[0], xyzs[1])
        if start_jump >= self.rmsd_threshold:
            raise WorkflowError(
                f"{label}: discontinuity at pathway start junction "
                f"(aligned RMSD = {start_jump:.4f} Å >= {self.rmsd_threshold} Å). "
                "IRC segment orientation may be flipped relative to optimized endpoints."
            )
        end_jump = _drms_aligned(xyzs[-2], xyzs[-1])
        if end_jump >= self.rmsd_threshold:
            raise WorkflowError(
                f"{label}: discontinuity at pathway end junction "
                f"(aligned RMSD = {end_jump:.4f} Å >= {self.rmsd_threshold} Å). "
                "IRC segment orientation may be flipped relative to optimized endpoints."
            )

    def _endpoint_rmsds(self, ep0, ep1, target_a, target_b):
        return {
            ("ep0", "a"): _drms_aligned(ep0, target_a),
            ("ep0", "b"): _drms_aligned(ep0, target_b),
            ("ep1", "a"): _drms_aligned(ep1, target_a),
            ("ep1", "b"): _drms_aligned(ep1, target_b),
        }

    def _matches(self, dist: float) -> bool:
        return dist < self.rmsd_threshold

    def _verify_irc_endpoints_distinct(self, ep0, ep1, label: str) -> None:
        """Raise if both IRC endpoints collapsed to the same structure."""
        _verify_irc_endpoints_distinct(
            ep0,
            ep1,
            rmsd_threshold=self.rmsd_threshold,
            label=label,
        )

    def _both_endpoints_match(self, ep0, ep1, target_a, target_b) -> Optional[bool]:
        """Return False/True for flipped/direct both-match; None if not both matched."""
        d = self._endpoint_rmsds(ep0, ep1, target_a, target_b)
        direct = self._matches(d[("ep0", "a")]) and self._matches(d[("ep1", "b")])
        flipped = self._matches(d[("ep0", "b")]) and self._matches(d[("ep1", "a")])
        if direct:
            return False
        if flipped:
            return True
        return None

    def _single_endpoint_match(self, ep0, ep1, target_a, target_b):
        """Return (ep_name, target_name) for a unique match, else None."""
        d = self._endpoint_rmsds(ep0, ep1, target_a, target_b)
        hits = []
        for ep_name in ("ep0", "ep1"):
            for t_name, t_xyz in (("a", target_a), ("b", target_b)):
                if self._matches(d[(ep_name, t_name)]):
                    hits.append((ep_name, t_name))
        if len(hits) == 1:
            return hits[0]
        return None

    def _dynamic_targets_after_single_match(
        self,
        ep0: np.ndarray,
        ep1: np.ndarray,
        target_a: np.ndarray,
        target_b: np.ndarray,
        ep_name: str,
        t_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        After one IRC end matches a dynamic target, move that label to the other IRC end.

        The unmatched label keeps its current target (initially the original global
        minimum from ``optimized_endpoints.xyz`` until it is matched in a later step).
        """
        other = ep1.copy() if ep_name == "ep0" else ep0.copy()
        next_a = np.asarray(target_a, dtype=float).reshape(-1, 3).copy()
        next_b = np.asarray(target_b, dtype=float).reshape(-1, 3).copy()
        if t_name == "a":
            next_a = other
        else:
            next_b = other
        return next_a, next_b

    def _register_ts(self, ts_mol) -> None:
        ts_xyz = _frame_xyz(ts_mol)
        for prev in self._discovered_ts:
            if _drms_aligned(prev, ts_xyz) < self.rmsd_threshold:
                raise WorkflowError(
                    "Duplicate transition-state structure encountered; stopping workflow."
                )
        self._discovered_ts.append(ts_xyz.copy())

    def _try_load_cached_elementary_result(self, paths, start_mol, end_mol):
        """Load a completed elementary IRC segment when on-disk artifacts are still valid."""
        endpoints_xyz = paths["endpoints_xyz"]
        irc_dir = paths["irc_dir"]
        start_xyz = _frame_xyz(start_mol)
        end_xyz = _frame_xyz(end_mol)
        if not _two_frame_xyz_matches(endpoints_xyz, start_xyz, end_xyz, self._GeoM):
            return None

        irc_log = irc_dir / _prefix_log(self.irc_prefix)
        irc_traj = irc_dir / _prefix_irc_traj(self.irc_prefix)
        if not (irc_log.is_file() and irc_traj.is_file() and _check_log_for_caching(irc_log)):
            return None

        try:
            traj, energies = _load_irc_from_run_dir(irc_dir, self._GeoM, self.irc_prefix)
            step_label = paths["label"]
            _verify_irc_endpoints_distinct(
                traj.xyzs[0],
                traj.xyzs[-1],
                rmsd_threshold=self.rmsd_threshold,
                label=step_label,
            )
            endpoints_path = irc_dir / "irc_endpoints.xyz"
            postopt_dir = irc_dir / "postopt"
            if endpoints_path.is_file() and postopt_dir.is_dir():
                reactant_opt = _load_frame_opt_trajectory(postopt_dir / "frame_0", self._GeoM)
                product_opt = _load_frame_opt_trajectory(postopt_dir / "frame_1", self._GeoM)
                endpoints = self._GeoM()
                endpoints.elem = list(traj.elem)
                endpoints.xyzs = [
                    reactant_opt.xyzs[-1].copy(),
                    product_opt.xyzs[-1].copy(),
                ]
                _verify_irc_endpoints_distinct(
                    endpoints.xyzs[0],
                    endpoints.xyzs[1],
                    rmsd_threshold=self.rmsd_threshold,
                    label=step_label,
                )
                result_traj, result_energies = _build_postopt_irc_trajectory(
                    reactant_opt,
                    traj,
                    product_opt,
                    energies,
                    self.rmsd_threshold,
                    self._GeoM,
                )
                _write_postopt_irc_trajectory(result_traj, irc_dir, self.irc_prefix)
                return _irc_result_dict(result_traj, result_energies, endpoints)
            endpoints = _irc_endpoints_molecule(traj, self._GeoM, energies)
            return _irc_result_dict(traj, energies, endpoints)
        except WorkflowError:
            raise
        except Exception:
            return None

    def _run_elementary(self, start_mol, end_mol, step_id: int):
        paths = self._step_artifact_paths(step_id)
        name = paths["label"]
        endpoints_xyz = paths["endpoints_xyz"]
        interp_xyz = paths["interp_xyz"]
        neb_dir = paths["neb_dir"]
        ts_dir = paths["ts_dir"]
        irc_dir = paths["irc_dir"]

        with _log_context(step=name):
            endpoints_xyz.parent.mkdir(parents=True, exist_ok=True)

            self._write_two_frame_xyz_if_changed(
                endpoints_xyz, start_mol, end_mol, f"{name} start", f"{name} end",
            )

            cached_result = self._try_load_cached_elementary_result(paths, start_mol, end_mol)
            if cached_result is not None:
                _log("Reusing cached elementary step (IRC + post-opt)", level=0)
                return cached_result

            step_dir = endpoints_xyz.parent
            interp_kwargs = self._calc_kwargs("interp", log_prefix="interpolate")
            log_prefix = str(interp_kwargs.pop("log_prefix", "interpolate"))
            n_images_interp = max(int(interp_kwargs.pop("n_images", self.interp_n_images)), 5)
            reuse_interp = _interpolation_cache_hit(
                step_dir,
                endpoints_xyz,
                n_images_interp,
                self._GeoM,
                log_prefix=log_prefix,
            )
            if reuse_interp:
                _log("Reusing interpolated chain", level=1)
                _ensure_canonical_interpolated_xyz(interp_xyz, step_dir)
            else:
                _log(f"TRICS interpolating ({n_images_interp} images)...", level=0)
                interpolated = interpolate(
                    str(endpoints_xyz),
                    run_dir=str(step_dir),
                    n_images=n_images_interp,
                    log_prefix=log_prefix,
                    **interp_kwargs,
                )
                interpolated.write(str(interp_xyz))
                _log(f"Wrote {len(interpolated.xyzs)} TRICS interpolation frames", level=0)

            neb_dir.mkdir(parents=True, exist_ok=True)
            _log("NEB running", level=0)
            neb_kwargs = self._calc_kwargs("neb")
            n_images_neb = int(neb_kwargs.pop("n_images", self.neb["n_images"]))
            neb_result = run_neb(
                str(self.input_file),
                str(interp_xyz),
                qm_program=self.qm_program,
                n_images=n_images_neb,
                nt=self.nt,
                run_dir=str(neb_dir),
                **neb_kwargs,
            )
            _log("NEB complete", level=0)

            ts_climb = neb_dir / f"{self.neb_prefix}.tsClimb.xyz"
            if not ts_climb.is_file() or neb_result.get("ts_guess") is None:
                _log(
                    "No climbing image from NEB; using converged chain as pathway",
                    level=0,
                )
                chain = _copy_trajectory(neb_result["optimized_chain"], self._GeoM)
                chain = self._orient_trajectory(
                    chain, _frame_xyz(start_mol), _frame_xyz(end_mol),
                )
                energies = list(getattr(chain, "qm_energies", None) or [])
                endpoints = _neb_endpoints_molecule(chain, self._GeoM, energies=energies or None)
                return _irc_result_dict(chain, energies, endpoints)

            _log("TS optimization running", level=0)
            ts_mol, ts_energy = optimize_ts(
                str(self.input_file),
                str(ts_climb),
                qm_program=self.qm_program,
                nt=self.nt,
                run_dir=str(ts_dir),
                **self._calc_kwargs("ts"),
            )
            self._register_ts(ts_mol)
            if ts_energy is not None:
                _log(f"TS energy = {ts_energy:.8f} Ha", level=0)
            else:
                _log("TS optimization complete", level=0)

            ts_optim = ts_dir / f"{self.ts_prefix}_optim.xyz"
            ts_hessian = ts_dir / f"{self.ts_prefix}.tmp" / "hessian" / "hessian.txt"
            if not ts_optim.is_file():
                raise WorkflowError(f"{name}: {ts_optim.name} not found after TS opt.")

            _log("IRC running", level=0)
            irc_result = run_irc(
                str(self.input_file),
                str(ts_optim),
                qm_program=self.qm_program,
                hessian=str(ts_hessian),
                nt=self.nt,
                run_dir=str(irc_dir),
                postopt=True,
                postopt_rmsd_threshold=self.rmsd_threshold,
                **self._irc_run_kwargs(),
            )
            return irc_result

    def _solve(self, start_mol, end_mol, target_a, target_b, depth: int):
        if depth >= self.max_depth:
            raise WorkflowError(
                f"Maximum workflow depth ({self.max_depth}) exceeded."
            )

        step_id = self._step_counter
        self._step_counter += 1
        irc_result = self._run_elementary(start_mol, end_mol, step_id)

        trj = irc_result["trj"]
        endpoints = irc_result["endpoints"]
        ep0, ep1 = endpoints.xyzs[0], endpoints.xyzs[1]
        # Use post-optimized minima as recursive connection points.
        conn0, conn1 = ep0, ep1

        step_label = self._step_artifact_paths(step_id)["label"]
        d = self._endpoint_rmsds(ep0, ep1, target_a, target_b)
        with _log_context(step=step_label):
            _log(
                "endpoint RMSDs (Å): "
                f"ep0-a={d[('ep0','a')]:.4f}, ep0-b={d[('ep0','b')]:.4f}, "
                f"ep1-a={d[('ep1','a')]:.4f}, ep1-b={d[('ep1','b')]:.4f}",
                level=0,
            )
        self._verify_irc_endpoints_distinct(ep0, ep1, step_label)

        both = self._both_endpoints_match(ep0, ep1, target_a, target_b)
        if both is not None:
            oriented = self._orient_trajectory(trj, target_a, target_b)
            if both:
                _log(
                    "IRC endpoints match optimized targets in flipped order "
                    f"(ep0→B, ep1→A); oriented pathway A → B",
                    level=0,
                )
            return oriented

        single = self._single_endpoint_match(ep0, ep1, target_a, target_b)
        if single is not None:
            ep_name, t_name = single
            missing_target = target_b if t_name == "a" else target_a
            missing_mol = end_mol if t_name == "a" else start_mol
            matched_target = target_a if t_name == "a" else target_b
            next_target_a, next_target_b = self._dynamic_targets_after_single_match(
                ep0, ep1, target_a, target_b, ep_name, t_name,
            )
            other_ep = "ep1" if ep_name == "ep0" else "ep0"
            with _log_context(step=step_label):
                _log(
                    f"Single match {ep_name}→{t_name}; "
                    f"dynamic target '{t_name}' set to {other_ep} (other IRC end), "
                    f"unmatched label unchanged; continuing pathway discovery",
                    level=0,
                )

            if ep_name == "ep0" and t_name == "b":
                junction = conn1
                core = self._orient_trajectory(trj, junction, matched_target)
                ext = self._solve(
                    missing_mol,
                    self._single_frame_mol(junction),
                    next_target_a,
                    next_target_b,
                    depth + 1,
                )
                return self._concat_segments([ext, core])
            if ep_name == "ep1" and t_name == "a":
                junction = conn0
                core = self._orient_trajectory(trj, matched_target, junction)
                ext = self._solve(
                    self._single_frame_mol(junction),
                    missing_mol,
                    next_target_a,
                    next_target_b,
                    depth + 1,
                )
                return self._concat_segments([core, ext])
            if ep_name == "ep0" and t_name == "a":
                junction = conn1
                core = self._orient_trajectory(trj, matched_target, junction)
                ext = self._solve(
                    self._single_frame_mol(junction),
                    missing_mol,
                    next_target_a,
                    next_target_b,
                    depth + 1,
                )
                return self._concat_segments([core, ext])
            if ep_name == "ep1" and t_name == "b":
                junction = conn0
                core = self._orient_trajectory(trj, junction, matched_target)
                ext = self._solve(
                    missing_mol,
                    self._single_frame_mol(junction),
                    next_target_a,
                    next_target_b,
                    depth + 1,
                )
                return self._concat_segments([ext, core])

        # No endpoint matches: orient via distinguishing PIC overlaps, then refine.
        mid, flipped = self._orient_trajectory_to_targets(trj, ep0, ep1, target_a, target_b)
        if flipped:
            ep0_target, ep1_target = target_b, target_a
            missing_start = not self._matches(d[("ep0", "b")])
            missing_end = not self._matches(d[("ep1", "a")])
            start_mol_ref = end_mol
            end_mol_ref = start_mol
        else:
            ep0_target, ep1_target = target_a, target_b
            missing_start = not self._matches(d[("ep0", "a")])
            missing_end = not self._matches(d[("ep1", "b")])
            start_mol_ref = start_mol
            end_mol_ref = end_mol

        raw_start, raw_end = mid.xyzs[0], mid.xyzs[-1]
        segments = []

        if missing_start:
            left = self._solve(
                start_mol_ref,
                self._single_frame_mol(raw_start),
                ep0_target,
                raw_start,
                depth + 1,
            )
            segments.append(left)

        segments.append(mid)

        if missing_end:
            right = self._solve(
                self._single_frame_mol(raw_end),
                end_mol_ref,
                raw_end,
                ep1_target,
                depth + 1,
            )
            segments.append(right)

        return self._concat_segments(segments)


def get_energies(
    input_file: Union[str, Path],
    xyz_file: Union[str, Path],
    *,
    qm_program: str = "psi4",
    nt: Optional[int] = None,
    work_dir: Optional[Union[str, Path]] = None,
    verbose: int = 0,
    stop_on_error: bool = False,
    **engine_kwargs,
) -> List[Optional[float]]:
    """
    Evaluate QM single-point energies (Hartree) for each frame in *xyz_file*.

    Uses geomeTRIC's ``get_molecule_engine`` and ``engine.calc_new``.

    Parameters
    ----------
    input_file : str or Path
        QM input template (method, basis, charge, multiplicity, etc.).
    xyz_file : str or Path
        Multi-frame XYZ trajectory.
    qm_program : str
        geomeTRIC engine name (default ``psi4``).
    nt : int, optional
        Number of threads passed to the engine when supported.
    work_dir : str or Path, optional
        Directory for per-frame scratch files (``energy_run/``). Defaults to cwd.
    verbose : int
        Logging verbosity for TRICFlow messages.
    stop_on_error : bool
        If True, raise ``WorkflowError`` on the first frame that fails.
        Otherwise failed frames are recorded as ``None`` and evaluation continues.
    **engine_kwargs
        Forwarded to ``get_molecule_engine`` when recognized.

    Returns
    -------
    list[float | None]
        Per-frame energies in Hartree, in trajectory order.
    """
    from geometric.prepare import get_molecule_engine
    from geometric.nifty import ang2bohr

    input_path = Path(input_file)
    xyz_path = Path(xyz_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"QM input file not found: {input_path}")
    if not xyz_path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for get_energies.")

    trajectory = GeoM(str(xyz_path))
    n_frames = len(getattr(trajectory, "xyzs", []) or [])
    if n_frames == 0:
        raise WorkflowError(f"XYZ file contains no frames: {xyz_path}")

    engine_setup: Dict[str, Any] = {
        "input": str(input_path),
        "engine": qm_program,
    }
    if nt is not None:
        engine_setup["nt"] = nt
    engine_setup.update(engine_kwargs)

    scratch_root = Path(work_dir).resolve() if work_dir is not None else Path.cwd().resolve()
    scratch_root = scratch_root / "energy_run"
    scratch_root.mkdir(parents=True, exist_ok=True)

    with _log_context(verbosity=verbose):
        _log(
            f"Evaluating energies for {n_frames} frames from {xyz_path.name} "
            f"(engine={qm_program})...",
            level=0,
        )
        _, engine = get_molecule_engine(**engine_setup)

        energies: List[Optional[float]] = []
        for frame_idx, geom in enumerate(trajectory):
            scratch_dir = scratch_root / f"frame_{frame_idx:04d}"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            try:
                coords = np.asarray(geom.xyzs)
                if coords.ndim == 3:
                    coords = coords[0]
                coords_bohr = coords.flatten() * ang2bohr
                result = engine.calc_new(coords_bohr, str(scratch_dir))
                energy = result.get("energy")
                if energy is None:
                    raise WorkflowError(
                        f"Engine returned no energy for frame {frame_idx}."
                    )
                energy = float(energy)
                energies.append(energy)
                _log(f"frame {frame_idx}: {energy:.8f} Ha", level=1)
            except Exception as exc:
                if stop_on_error:
                    raise WorkflowError(
                        f"Energy evaluation failed for frame {frame_idx}: {exc}"
                    ) from exc
                _log_warn(
                    f"frame {frame_idx}: energy evaluation failed ({exc})"
                )
                energies.append(None)

        _log(f"Energy evaluation complete ({len(energies)} frames)", level=0)
        return energies


def _oriented_segment_copy(mol, reversed_path: bool, GeoM):
    """Return a trajectory copy, optionally reversed."""
    if reversed_path:
        return _reverse_trajectory(mol, GeoM)
    return _copy_trajectory(mol, GeoM)


def _segments_connect(
    exit_xyz: np.ndarray,
    entry_xyz: np.ndarray,
    rmsd_threshold: float,
) -> bool:
    return _drms_aligned(exit_xyz, entry_xyz) <= rmsd_threshold


def _build_assembly_graph(
    segments: Sequence[Any],
    *,
    rmsd_threshold: float,
    GeoM,
) -> Tuple[List[set], Dict[int, List[Tuple[int, bool, bool, float]]]]:
    """
    Build undirected connected components and directed adjacency for segment assembly.

    Adjacency maps segment index → list of ``(next_idx, rev_self, rev_next, rmsd)``.
    """
    n = len(segments)
    undirected: List[set] = [set() for _ in range(n)]
    directed: Dict[int, List[Tuple[int, bool, bool, float]]] = {i: [] for i in range(n)}

    for i in range(n):
        for rev_i in (False, True):
            mol_i = _oriented_segment_copy(segments[i], rev_i, GeoM)
            exit_i = mol_i.xyzs[-1]
            entry_i = mol_i.xyzs[0]
            for j in range(n):
                if i == j:
                    continue
                for rev_j in (False, True):
                    mol_j = _oriented_segment_copy(segments[j], rev_j, GeoM)
                    entry_j = mol_j.xyzs[0]
                    exit_j = mol_j.xyzs[-1]
                    cost_fwd = _drms_aligned(exit_i, entry_j)
                    if cost_fwd <= rmsd_threshold:
                        directed[i].append((j, rev_i, rev_j, float(cost_fwd)))
                        undirected[i].add(j)
                        undirected[j].add(i)
                    cost_rev = _drms_aligned(exit_j, entry_i)
                    if cost_rev <= rmsd_threshold:
                        directed[j].append((i, rev_j, rev_i, float(cost_rev)))
                        if j not in undirected[i]:
                            undirected[i].add(j)
                            undirected[j].add(i)

    visited: set = set()
    components: List[set] = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(undirected[node] - visited)
        components.append(comp)
    return components, directed


def _longest_assembly_chain(
    component: set,
    directed: Dict[int, List[Tuple[int, bool, bool, float]]],
    *,
    rmsd_threshold: float,
) -> Tuple[List[int], Dict[int, bool]]:
    """Return the longest connectable chain and per-segment orientation within *component*."""
    best_order: List[int] = []
    best_orient: Dict[int, bool] = {}

    def dfs(path: List[int], orients: Dict[int, bool]) -> None:
        nonlocal best_order, best_orient
        if len(path) > len(best_order):
            best_order = path[:]
            best_orient = dict(orients)
        last = path[-1]
        rev_last = orients[last]
        candidates: List[Tuple[float, int, bool]] = []
        for nxt, rev_self, rev_next, cost in directed.get(last, []):
            if nxt not in component or nxt in path:
                continue
            if rev_self != rev_last:
                continue
            candidates.append((cost, nxt, rev_next))
        candidates.sort()
        for _, nxt, rev_next in candidates:
            orients[nxt] = rev_next
            dfs(path + [nxt], orients)
            del orients[nxt]

    for start in sorted(component):
        for rev_start in (False, True):
            dfs([start], {start: rev_start})

    return best_order, best_orient


def _cover_component_with_chains(
    component: set,
    directed: Dict[int, List[Tuple[int, bool, bool, float]]],
    *,
    rmsd_threshold: float,
) -> List[Tuple[List[int], Dict[int, bool]]]:
    """Greedily extract longest chains until every segment in *component* is assigned."""
    remaining = set(component)
    chains: List[Tuple[List[int], Dict[int, bool]]] = []
    while remaining:
        order, orients = _longest_assembly_chain(
            remaining, directed, rmsd_threshold=rmsd_threshold,
        )
        if len(order) <= 1 and len(remaining) > 1:
            order = [min(remaining)]
            orients = {order[0]: False}
        elif len(order) <= 1 and len(remaining) == 1:
            order = [next(iter(remaining))]
            orients = {order[0]: False}
        chains.append((order, orients))
        remaining -= set(order)
    return chains


def _concat_oriented_segments(
    order: Sequence[int],
    orientations: Dict[int, bool],
    segments: Sequence[Any],
    *,
    rmsd_threshold: float,
    GeoM,
):
    """Concatenate oriented segments in *order*."""
    if not order:
        raise WorkflowError("Cannot assemble an empty segment list.")
    pathway = _oriented_segment_copy(segments[order[0]], orientations[order[0]], GeoM)
    for idx in order[1:]:
        nxt = _oriented_segment_copy(segments[idx], orientations[idx], GeoM)
        if not _segments_connect(pathway.xyzs[-1], nxt.xyzs[0], rmsd_threshold):
            raise WorkflowError(
                f"Segment chain is not connected at junction index "
                f"{order.index(idx) - 1} → {order.index(idx)}."
            )
        pathway = _concat_trajectories(
            pathway, nxt, pathway.xyzs[-1], rmsd_threshold, GeoM,
        )
    return pathway


def assemble_pathways(
    xyz_files: Sequence[Union[str, Path]],
    *,
    rmsd_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Assemble one or more pathways from multi-frame XYZ segments.

    Segments are oriented and concatenated when junction RMSDs are below
    *rmsd_threshold*. When not all inputs connect into one chain, returns the
    longest possible chains within each connected component.

    Parameters
    ----------
    xyz_files : sequence of str or Path
        Multi-frame XYZ trajectories to assemble (order among inputs is not
        assumed; connectivity is inferred from junction RMSDs).
    rmsd_threshold : float
        Aligned RMSD threshold (Å) for declaring that two segment ends connect.

    Returns
    -------
    list[dict]
        One entry per assembled pathway with keys ``pathway`` (Molecule),
        ``segment_names``, ``segment_order``, ``orientations``, and ``n_frames``.
    """
    GeoM = _get_geo_molecule()
    if GeoM is None:
        raise RuntimeError("geomeTRIC is required for assemble_pathways.")

    if not xyz_files:
        raise ValueError("assemble_pathways requires at least one XYZ file.")

    segments = []
    names = []
    for path in xyz_files:
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"Segment XYZ not found: {src}")
        mol = GeoM(str(src))
        if len(getattr(mol, "xyzs", []) or []) < 1:
            raise WorkflowError(f"Segment XYZ contains no frames: {src}")
        segments.append(mol)
        names.append(src.name)

    components, directed = _build_assembly_graph(
        segments, rmsd_threshold=rmsd_threshold, GeoM=GeoM,
    )

    results: List[Dict[str, Any]] = []
    for comp in sorted(components, key=lambda c: (-len(c), min(c))):
        for order, orients in _cover_component_with_chains(
            comp, directed, rmsd_threshold=rmsd_threshold,
        ):
            pathway = _concat_oriented_segments(
                order, orients, segments,
                rmsd_threshold=rmsd_threshold, GeoM=GeoM,
            )
            pathway.align()
            results.append({
                "pathway": pathway,
                "segment_names": [names[i] for i in order],
                "segment_order": list(order),
                "orientations": {names[i]: orients[i] for i in order},
                "n_frames": len(pathway.xyzs),
            })

    if not results:
        raise WorkflowError("No pathway segments could be assembled.")

    return results


def _cli_str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "yeah", "ok", "on", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "nope", "off", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {value!r}")


def _cli_add_qm_args(parser) -> None:
    parser.add_argument(
        "--input-file",
        required=True,
        help="QM input template (method/basis/charge/mult); geometry is overridden",
    )
    parser.add_argument("--qm-program", default="psi4", help="geomeTRIC engine (default: psi4)")
    parser.add_argument("--run-dir", default=None, help="Working directory for calculation artifacts")
    parser.add_argument("--nt", type=int, default=None, help="Pass --nt to geomeTRIC")
    parser.add_argument("--coordsys", default="tric", help="geomeTRIC coordinate system (default: tric)")


def main_optimize(argv=None):
    """CLI entry point for tricflow-optimize."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize one or more XYZ frames locally with geomeTRIC (TRICFlow).",
    )
    parser.add_argument("xyz_file", help="XYZ file (single frame or multi-frame trajectory)")
    _cli_add_qm_args(parser)
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args(argv)

    opt_kwargs = {"coordsys": args.coordsys}
    if args.nt is not None:
        opt_kwargs["nt"] = args.nt
    if args.maxiter is not None:
        opt_kwargs["maxiter"] = args.maxiter

    result = optimize_frames(
        args.input_file,
        args.xyz_file,
        qm_program=args.qm_program,
        run_dir=args.run_dir,
        **opt_kwargs,
    )
    n = len(result.xyzs) if hasattr(result, "xyzs") else 0
    _log(f"Optimized {n} frame(s).", level=0)
    return 0


def main_neb(argv=None):
    """CLI entry point for tricflow-neb."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NEB and optimize the climbing-image TS when one is found.",
    )
    parser.add_argument("initial_chain", help="Multi-frame XYZ initial chain for NEB")
    _cli_add_qm_args(parser)
    parser.add_argument("--ts-run-dir", default=None, help="TS optimization directory (default: <run-dir>/../ts_run)")
    parser.add_argument("--n-images", type=int, default=11, help="Number of NEB images")
    parser.add_argument(
        "--maxg",
        type=float,
        default=None,
        help="NEB convergence: max RMS gradient over all images (eV/Å; geomeTRIC default 0.05)",
    )
    parser.add_argument(
        "--avgg",
        type=float,
        default=None,
        help="NEB convergence: average RMS gradient over all images (eV/Å; geomeTRIC default 0.025)",
    )
    parser.add_argument("--neb-prefix", default="neb", help="geomeTRIC prefix for NEB artifacts")
    parser.add_argument("--ts-prefix", default="ts", help="geomeTRIC prefix for TS artifacts")
    parser.add_argument("--ts-converge", default=None, help="geomeTRIC --converge for TS optimization")
    args = parser.parse_args(argv)

    neb_kwargs: Dict[str, Any] = {"coordsys": args.coordsys, "prefix": args.neb_prefix}
    if args.n_images is not None:
        neb_kwargs["n_images"] = args.n_images
    if args.maxg is not None:
        neb_kwargs["maxg"] = args.maxg
    if args.avgg is not None:
        neb_kwargs["avgg"] = args.avgg

    ts_kwargs: Dict[str, Any] = {"coordsys": args.coordsys, "prefix": args.ts_prefix}
    if args.ts_converge is not None:
        ts_kwargs["converge"] = args.ts_converge

    result = run_neb_with_ts_opt(
        args.input_file,
        args.initial_chain,
        qm_program=args.qm_program,
        nt=args.nt,
        run_dir=args.run_dir,
        ts_run_dir=args.ts_run_dir,
        neb_kwargs=neb_kwargs,
        ts_kwargs=ts_kwargs,
    )
    if result["ts_xyz"] is not None:
        _log(f"TS optimization complete → {result['ts_xyz']}", level=0)
    else:
        _log("NEB complete (no TS climbing image).", level=0)
    return 0


def main_irc(argv=None):
    """CLI entry point for tricflow-irc."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run IRC from an optimized TS and post-optimize IRC endpoints.",
    )
    parser.add_argument("ts_xyz", help="Optimized TS XYZ (e.g. ts_optim.xyz)")
    _cli_add_qm_args(parser)
    parser.add_argument(
        "--hessian",
        default=None,
        help="Reuse an existing Hessian file (default: calculate at TS geometry)",
    )
    parser.add_argument("--trust", type=float, default=None, help="IRC trust radius")
    parser.add_argument("--irc-prefix", default="irc", help="geomeTRIC prefix for IRC artifacts")
    args = parser.parse_args(argv)

    irc_kwargs: Dict[str, Any] = {"coordsys": args.coordsys, "prefix": args.irc_prefix}
    if args.trust is not None:
        irc_kwargs["trust"] = args.trust

    result = run_irc_postopt(
        args.input_file,
        args.ts_xyz,
        qm_program=args.qm_program,
        hessian=args.hessian,
        nt=args.nt,
        run_dir=args.run_dir,
        **irc_kwargs,
    )
    postopt_xyz = result.get("postopt_xyz")
    if postopt_xyz is not None:
        _log(f"Post-opt IRC pathway → {postopt_xyz}", level=0)
    return 0


def main_tsoptimize(argv=None):
    """CLI entry point for tricflow-tsoptimize."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a transition-state guess (hessian first+last); optional IRC follow-up.",
    )
    parser.add_argument("ts_guess", help="TS guess XYZ (e.g. neb.tsClimb.xyz)")
    _cli_add_qm_args(parser)
    parser.add_argument(
        "--postirc",
        type=_cli_str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="yes|no",
        help='When "yes", run IRC with endpoint post-optimization if exactly 1 imaginary mode is found',
    )
    parser.add_argument("--irc-run-dir", default=None, help="IRC directory (default: <run-dir>/../irc_run)")
    parser.add_argument("--ts-prefix", default="ts", help="geomeTRIC prefix for TS artifacts")
    parser.add_argument("--irc-prefix", default="irc", help="geomeTRIC prefix for IRC artifacts")
    parser.add_argument("--converge", default=None, help="geomeTRIC --converge for TS optimization")
    parser.add_argument("--trust", type=float, default=None, help="IRC trust radius when --postirc yes")
    args = parser.parse_args(argv)

    ts_kwargs: Dict[str, Any] = {"coordsys": args.coordsys, "prefix": args.ts_prefix}
    if args.converge is not None:
        ts_kwargs["converge"] = args.converge

    irc_kwargs: Dict[str, Any] = {"coordsys": args.coordsys, "prefix": args.irc_prefix}
    if args.trust is not None:
        irc_kwargs["trust"] = args.trust

    result = optimize_ts_with_postirc(
        args.input_file,
        args.ts_guess,
        postirc=args.postirc,
        qm_program=args.qm_program,
        nt=args.nt,
        run_dir=args.run_dir,
        irc_run_dir=args.irc_run_dir,
        ts_kwargs=ts_kwargs,
        irc_kwargs=irc_kwargs,
    )
    _log(f"TS optimization complete → {result['ts_xyz']}", level=0)
    if result.get("n_imaginary_modes") is not None:
        _log(f"Imaginary modes: {result['n_imaginary_modes']}", level=0)
    if result.get("postopt_xyz") is not None:
        _log(f"Post-opt IRC pathway → {result['postopt_xyz']}", level=0)
    return 0


def main_get_energies(argv=None):
    """CLI entry point for tricflow-energies."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate QM single-point energies for each frame in an XYZ trajectory.",
    )
    parser.add_argument("xyz_file", help="Multi-frame XYZ trajectory")
    _cli_add_qm_args(parser)
    parser.add_argument(
        "-o",
        "--output",
        default="energies.json",
        help="Output JSON file (default: energies.json)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on the first frame that fails (default: record null and continue)",
    )
    args = parser.parse_args(argv)

    energies = get_energies(
        args.input_file,
        args.xyz_file,
        qm_program=args.qm_program,
        nt=args.nt,
        work_dir=args.run_dir,
        stop_on_error=args.stop_on_error,
    )
    out_path = Path(args.output)
    out_path.write_text(json.dumps({"energies": energies}, indent=4))
    n_ok = sum(e is not None for e in energies)
    _log(f"{n_ok}/{len(energies)} frames evaluated → {out_path}", level=0)
    return 0


def main_assemble(argv=None):
    """CLI entry point for tricflow-assemble."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Assemble multi-frame XYZ segments into one or more connected pathways.",
    )
    parser.add_argument(
        "xyz_files",
        nargs="+",
        help="Trajectory segments to assemble (connectivity inferred from junction RMSDs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="full_pathway.xyz",
        help="Output XYZ for a single assembled pathway (default: full_pathway.xyz)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for assembled pathway XYZ files",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.1,
        help="Aligned RMSD threshold (Å) for segment junctions (default: 0.1)",
    )
    args = parser.parse_args(argv)

    results = assemble_pathways(
        args.xyz_files,
        rmsd_threshold=args.rmsd_threshold,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    suffix = output.suffix or ".xyz"
    stem = output.stem or "full_pathway"

    if len(results) == 1:
        out_path = out_dir / output.name
        results[0]["pathway"].align()
        results[0]["pathway"].write(str(out_path))
        names = " → ".join(results[0]["segment_names"])
        _log(
            f"Assembled {results[0]['n_frames']} frames from {len(results[0]['segment_names'])} "
            f"segment(s) ({names}) → {out_path}",
            level=0,
        )
    else:
        for idx, item in enumerate(results, start=1):
            out_path = out_dir / f"{stem}_{idx:02d}{suffix}"
            item["pathway"].align()
            item["pathway"].write(str(out_path))
            names = " → ".join(item["segment_names"])
            _log(
                f"Pathway {idx}: {item['n_frames']} frames "
                f"({len(item['segment_names'])} segment(s): {names}) → {out_path}",
                level=0,
            )
        _log(f"Wrote {len(results)} pathway(s) (inputs do not form one connected chain).", level=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_optimize())

__all__ = [
    "TRICWorkflow",
    "optimize_frames",
    "interpolate",
    "run_neb",
    "run_neb_with_ts_opt",
    "optimize_ts",
    "optimize_ts_with_postirc",
    "run_irc",
    "run_irc_postopt",
    "get_energies",
    "assemble_pathways",
    "main_optimize",
    "main_neb",
    "main_irc",
    "main_tsoptimize",
    "main_get_energies",
    "main_assemble",
]
