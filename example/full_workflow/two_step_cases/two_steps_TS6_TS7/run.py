from pathlib import Path

from tricflow import TRICWorkflow

here = Path(__file__).parent

workflow = TRICWorkflow(
    here / "psi4.in",
    work_dir=here,
    qm_program="psi4",
    nt=4,
    rmsd_threshold=0.1,
    max_depth=10,
    neb={"n_images": 17, "maxg": 0.05, "avgg": 0.025, "coordsys": "tric"},
    opt={"coordsys": "tric"},
    ts={},
    irc={"coordsys": "tric"},
    interp={"n_images": 50},
)

pathway = workflow.run(here / "initial.xyz")
print(f"Done. Full pathway has {len(pathway.xyzs)} frames.")
