"""Apps: a ComfyUI workflow plus a manifest that turns it into an interface.

    manifest.py  load + validate app.json (bindings are checked against the workflows)
    compile.py   inputs (+ sweeps) -> one API-format prompt per grid cell
    runtime.py   submit a batch to ComfyUI, track it, collect outputs

See docs/APPS.md for the design.
"""
