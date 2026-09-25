import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from wigglystuff import PlaySlider

    from scm.moliterate import load_dataset
    from scm.moliterate.visualize.structure_viewer import render_structure_html

    return PlaySlider, load_dataset, mo, np, render_structure_html


@app.cell
def _(mo):
    path = mo.ui.text(
        label="Structure path",
        placeholder="Paste a .xyz, .extxyz, .cif, .db, .rkf, or ParAMS path",
        kind="text",
        full_width=True,
        value=(
            "/home/bene/Documents/work/ALIR/Code/active_learning_workspace/active_learning/ALruns/"
            "20260326_113420/iter_01/M3GNet00-OCC=O-AMSMDTask/ams.rkf"
        ),
    )
    return (path,)


@app.cell
def _(mo, path):
    mo.vstack(
        [
            mo.md(
                """
                # Moliterate Structure Viewer

                Paste a path to a structure file or Moliterate-supported dataset.
                The viewer will load all frames, expose a slider, and render guessed bonds as sticks.
                """
            ),
            path,
        ]
    )
    return


@app.cell
def _(load_dataset, mo, path):
    path_value = path.value.strip()
    try:
        ds = load_dataset(path_value)
    except Exception as e:
        mo.stop(True, mo.md(f"Raised an Error during data retrival: {e}"))
    return (ds,)


@app.cell
def _(PlaySlider, ds, mo):
    frame_slider = mo.ui.anywidget(
        PlaySlider(
            min_value=0, max_value=len(ds) - 1, step=1, interval_ms=100, loop=True, show_value=True, debounce=True
        )
    )

    # frame_slider = mo.ui.slider(0, len(ds) - 1, value=0, step=1, label="Frame", show_value=True, debounce=True)
    return (frame_slider,)


@app.function
def preprocess_atoms(atoms):
    atoms.center()
    return atoms


@app.cell
def _(ds, frame_slider, mo, render_structure_html):
    entry = ds[int(frame_slider.value["value"])]
    slider_plot = mo.vstack(
        [
            frame_slider,
            mo.iframe(render_structure_html(preprocess_atoms(entry.atoms), width=800), height="500", width="1000"),
        ]
    )
    slider_plot
    return (entry,)


@app.cell
def _(entry, mo):
    mo.ui.table([{"key": k, "value": str(v)} for k, v in entry.metadata.items()])
    return


@app.cell
def _(entry, mo, np):
    mo.ui.table([{"property": k, "max-component": np.asarray(v).max().round(5)} for k, v in entry.properties.items()])
    return


if __name__ == "__main__":
    app.run()
