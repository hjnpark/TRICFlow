from tricflow import assemble_pathways

results = assemble_pathways(["TS3_path.xyz", "TS4_path.xyz", "TS5_path.xyz", "TS6_path.xyz"], rmsd_threshold=0.1)
pathway = results[0]["pathway"]          # geomeTRIC Molecule
results[0]["segment_names"]              # assembly order
results[0]["orientations"]              # which segments were reversed
results[0]["n_frames"]    

pathway.write('full_pathway.xyz')


