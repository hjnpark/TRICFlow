#!/usr/bin/env python3
"""Simple HCN workflow: optimize → interpolate → NEB → TS opt → IRC."""

from pathlib import Path

from tricflow import interpolate, optimize_frames, optimize_ts, run_irc, run_neb

here = Path(__file__).parent
input_file = here / "psi4_HCN.in"
initial_xyz = here / "input_HCN.xyz"

# 1. Optimize the two endpoint frames
opt_mols = optimize_frames(
    str(input_file),
    str(initial_xyz),
    qm_program="psi4",
    run_dir=str(here / "opt_runs"),
    nt=4,
    coordsys="tric",
)
opt_mols.write(str(here / "optimized_endpoints.xyz"))

# 2. TRICS interpolation
step_dir = here / "step_00"
interpolated = interpolate(
    str(here / "optimized_endpoints.xyz"),
    run_dir=str(step_dir),
    n_images=50,
)
interpolated.write(str(step_dir / "interpolated.xyz"))

# 3. NEB
neb_dir = step_dir / "neb_run"
neb_result = run_neb(
    str(input_file),
    str(step_dir / "interpolated.xyz"),
    qm_program="psi4",
    run_dir=str(neb_dir),
    nt=4,
    n_images=11,
)
ts_climb = neb_dir / "psi4_HCN.tsClimb.xyz"

# 4. TS optimization (needed before IRC)
ts_dir = step_dir / "ts_run"
ts_mol, _ = optimize_ts(
    str(input_file),
    str(ts_climb),
    qm_program="psi4",
    run_dir=str(ts_dir),
    nt=4,
    coordsys="tric",
    converge="set GAU_TIGHT",
)
ts_optim = ts_dir / "psi4_HCN_optim.xyz"

# 5. IRC with post-optimization of endpoints
irc_dir = step_dir / "irc_run"
irc_result = run_irc(
    str(input_file),
    str(ts_optim),
    qm_program="psi4",
    run_dir=str(irc_dir),
    nt=4,
    coordsys="tric",
    trust=0.3,
    postopt=True,
)
pathway = irc_result.get("trj")
pathway.write("final_path.xyz")
print(f"Done. IRC pathway has {len(irc_result['trj'].xyzs)} frames.")
