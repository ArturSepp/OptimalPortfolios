"""
pytest rootdir marker for the replication package.

Its presence makes pytest's default (prepend) import mode add this folder to
sys.path during collection, so the flat replication modules
(local_path, governed_cma_projection, run_*) import inside tests exactly as
they do under `python <script>.py`. No sys.path mutation in package code.

Does not belong here: fixtures with computational content (the tests load the
snapshot themselves so a failure names the loader, not a fixture).
"""
