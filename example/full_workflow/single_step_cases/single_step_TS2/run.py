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
    neb={"n_images": 11, "maxg": 0.1, "avgg": 0.05, "plain":1},
    opt={},
    ts={},
    irc={},
    interp={"n_images": 50},
)

pathway = workflow.run(here / "initial.xyz")
print(f"Done. Full pathway has {len(pathway.xyzs)} frames.")
