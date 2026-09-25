from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import cycle
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union

from pydantic import BaseModel
from scm.moliterate.filters import ConditionsFilter, RemoveDuplicates
from scm.moliterate.partition_criteria import MetadataCriterion
from scm.moliterate.visualize import PropertyDistributionPlot
from tabulate import tabulate

from scm.active_learning.callbacks import FolderManagerCallback
from scm.active_learning.loop.loop import ActiveLearningLoop
from scm.active_learning.loop.loop_result import TableKind


def _check_analysis_dependencies():
    try:
        import pandas as pd  # noqa: F401
        import seaborn as sns  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "seaborn and pandas not found. To install extra components do pip install .[viz]"
        ) from e


class ActiveLearningAnalysis(BaseModel):
    loop: ActiveLearningLoop

    def collect_table(self, options: TableKind = "summary", sort_values=None, **kwargs):
        table = self.loop.result.collect_table(options=options, **kwargs)
        if sort_values is None:
            for x in table:
                yield x
            return

        if isinstance(sort_values, str):
            sort_values = (sort_values,)

        rows = list(table)
        rows.sort(key=itemgetter(*sort_values))
        for x in rows:
            yield x

    def view_table(
        self,
        options: Union[TableKind, Literal["history", "journey"]] = "summary",
        tabulate_kwargs=None,
        **kwargs,
    ):
        if options == "history":
            return self.loop.journey.history_table()
        if options == "journey":
            return self.loop.journey.journey_table()
        if tabulate_kwargs is None:
            tabulate_kwargs = {}
        return tabulate(
            self.collect_table(options=options, **kwargs),
            headers="keys",
            **tabulate_kwargs,
        )

    def get_logfile_path(self, idx: int = 0):
        folder_callbacks = self.loop.query_callbacks(FolderManagerCallback)
        if len(folder_callbacks) == 0:
            return None
        p = folder_callbacks[idx].logging_path
        if p and p.exists():
            return p

    @staticmethod
    def _get_group_sort_key(group: str, group_tpl: Iterable[str], join_key: str):
        if tuple(group_tpl) == ("dataset", "iteration_al"):
            splitted = group.split(join_key)
            iteration = splitted[1]
            if iteration == "None":
                return (splitted[0], 0, 0)
            return (splitted[0], 1, int(iteration))
        return group

    def _collect_grouped_rows(
        self,
        row_builder: Callable[[str, str, Any], dict[str, Any]],
        include_overall: bool = True,
        group_by_metadata: Iterable[Iterable[str]] | None = (
            ("dataset",),
            ("dataset", "iteration_al"),
        ),
        join_key: str = "%",
    ) -> Iterator[dict[str, Any]]:
        dataset = self.loop.splitting.dataset
        rows = []

        if include_overall:
            rows.append(row_builder("overall", "all", dataset))

        if group_by_metadata is not None:
            for group_tpl in group_by_metadata:
                criterion = MetadataCriterion(metadata_keys=list(group_tpl), concat_symbol=join_key)
                grouped = criterion.group_by(criterion(dataset))
                scope = join_key.join(group_tpl)
                for group, indices in sorted(
                    grouped.items(),
                    key=lambda item: self._get_group_sort_key(item[0], group_tpl, join_key),
                ):
                    rows.append(row_builder(scope, group, dataset[indices]))

        for row in rows:
            yield row

    def collect_train_validation_frames(
        self,
        include_overall: bool = False,
        group_by_metadata: Iterable[Iterable[str]] | None = (("iteration_al",),),
        join_key: str = "%",
    ):
        train_filter, valid_filter = self.loop.splitting.train_validation_filters

        def get_row(scope, group, subset):
            train_set = train_filter(subset)
            validation_set = valid_filter(subset)
            return {
                "scope": scope,
                "group": group,
                "n_frames_training": len(train_set),
                "n_frames_validation": len(validation_set),
                "tot_frames": len(subset),
            }

        yield from self._collect_grouped_rows(
            row_builder=get_row,
            include_overall=include_overall,
            group_by_metadata=group_by_metadata,
            join_key=join_key,
        )

    def collect_duplicates(
        self,
        option: str = "AgglXYZEuclKB",
        seed: Optional[int] = 22,
        include_overall: bool = True,
        group_by_metadata: Iterable[Iterable[str]] | None = (
            ("dataset",),
            ("dataset", "iteration_al"),
        ),
        join_key: str = "%",
    ):
        duplicate_filter = RemoveDuplicates.build(options=option, num_samples_by_group=1, seed=seed)

        def get_row(scope, group, tot_frames, n_frames_unique):
            return {
                "scope": scope,
                "group": group,
                "tot_frames": tot_frames,
                "n_frames_unique": n_frames_unique,
                "n_removed": tot_frames - n_frames_unique,
            }

        def build_row(scope, group, subset):
            return get_row(
                scope,
                group,
                len(subset),
                len(duplicate_filter(subset)),
            )

        yield from self._collect_grouped_rows(
            row_builder=build_row,
            include_overall=include_overall,
            group_by_metadata=group_by_metadata,
            join_key=join_key,
        )

    def view_duplicates(
        self,
        tabulate_kwargs=None,
        option: str = "AgglXYZEuclKB",
        seed: Optional[int] = 22,
        include_overall: bool = True,
        group_by_metadata: Iterable[Iterable[str]] | None = (
            ("dataset",),
            ("dataset", "iteration_al"),
        ),
    ):
        if tabulate_kwargs is None:
            tabulate_kwargs = {}
        return tabulate(
            self.collect_duplicates(
                option=option,
                seed=seed,
                include_overall=include_overall,
                group_by_metadata=group_by_metadata,
            ),
            headers="keys",
            **tabulate_kwargs,
        )

    @property
    def plot(self):
        return ALPlotter(self)

    def get_summary(self, options: Literal["convergence"] = "convergence"):
        train, _ = self.loop.splitting.train_validation_filters
        first_iter = ConditionsFilter(conditions="iteration_al<0", collect_from="metadata")
        options_call = {
            "convergence": lambda: {
                "Reason": self.loop.iterable_loop.reason,
                "NSteps": len(self.loop.result.iterations),
                "Ninit": len((train & first_iter)(self.loop.splitting.dataset)),
                "Nfinal": len(train(self.loop.splitting.dataset)),
                "Tot[min]": int(
                    sum([x.total_seconds() / 60 for xx in self.loop.result.iterations for x in xx.timings.values()])
                ),
            },
        }
        return options_call[options]()


class ALPlotter:
    """
    Simple class with some plotting utilities.
    The plots are meant to be served as example rather than reusable feature."""

    def __init__(self, an: ActiveLearningAnalysis):
        self.analysis = an

    def collection_plots(self, add_duplicates_check: bool = False):
        _check_analysis_dependencies()
        import matplotlib.pyplot as plt

        def fig_table():
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(
                0.01,
                0.99,
                self.analysis.view_duplicates(),
                family="monospace",
                va="top",
            )
            g = type("G", (), {})()
            g.figure = fig
            return g

        plot_methods = [
            self.summary_table,
            self.train_validation_frames,
            self.timings_table,
            self.dataset_property_distributions,
            self.validity_check_histogram,
            self.accuracy_check_histogram,
            self.train_check_histogram,
            self.summary_validity_table,
            self.validity_table,
            self.accuracy_table,
            self.train_table,
            self.train_check_table,
        ]
        if add_duplicates_check:
            plot_methods.append(fig_table)
        return plot_methods

    def save_to_pdf(self, name: str = "analysis.pdf", folder: str = "<ALFolder>"):
        _check_analysis_dependencies()
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        if folder == "<ALFolder>":
            folder_callbacks = self.analysis.loop.query_callbacks(FolderManagerCallback)
            if len(folder_callbacks) == 0:
                raise ValueError("Cannot resolve <ALFolder> without a FolderManagerCallback.")
            output_dir = folder_callbacks[0].run_dir()
        else:
            output_dir = Path(folder)

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / name

        with PdfPages(pdf_path) as pdf:
            for plot_method in self.collection_plots():
                try:
                    g = plot_method()
                except (AssertionError, ValueError):
                    continue
                pdf.savefig(g.figure, bbox_inches="tight")
                plt.close(g.figure)
        print(pdf_path)
        return pdf_path

    def _train_validation_frames_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(self.analysis.collect_train_validation_frames())
        assert len(df) > 0, "No train/validation frame data"

        if "scope" in df.columns:
            df = df[df["scope"] == "iteration_al"]
        assert len(df) > 0, "No per-iteration train/validation frame data"

        df = df.assign(iteration_al=pd.to_numeric(df["group"], errors="coerce")).dropna(subset=["iteration_al"])
        assert len(df) > 0, "No numeric iteration_al values for train/validation frame data"

        df = df.sort_values("iteration_al")
        return df.assign(
            total_frames_training=df["n_frames_training"].cumsum(),
            total_frames_validation=df["n_frames_validation"].cumsum(),
            total_frames=df["tot_frames"].cumsum(),
        )

    def train_validation_frames(self, figsize=None):
        _check_analysis_dependencies()
        import matplotlib.pyplot as plt

        df = self._train_validation_frames_dataframe()
        final_total = int(df["total_frames"].iloc[-1])
        final_training = int(df["total_frames_training"].iloc[-1])
        final_validation = int(df["total_frames_validation"].iloc[-1])

        fig, ax = plt.subplots(figsize=figsize or (8, 6))
        ax.plot(
            df["iteration_al"],
            df["total_frames"],
            marker="D",
            markersize=6.5,
            markeredgewidth=1.25,
            markeredgecolor="white",
            label=f"total (final: {final_total})",
        )
        ax.plot(
            df["iteration_al"],
            df["total_frames_training"],
            marker="o",
            markersize=7,
            markeredgewidth=1.25,
            markeredgecolor="white",
            label=f"training (final: {final_training})",
        )
        ax.plot(
            df["iteration_al"],
            df["total_frames_validation"],
            marker="s",
            markersize=6.5,
            markeredgewidth=1.25,
            markeredgecolor="white",
            label=f"validation (final: {final_validation})",
        )

        ax.set_title("Training and Validation Dataset Frames")
        ax.set_xlabel("iteration_al")
        ax.set_ylabel("Total number of frames")
        ax.grid(which="both", ls="--", linewidth=1)
        ax.set_axisbelow(True)
        ax.legend()
        fig.tight_layout()

        g = type("G", (), {})()
        g.figure = fig
        return g

    def summary_table(self, figsize=None):
        _check_analysis_dependencies()
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(self.analysis.collect_table("summary"))
        assert len(df) > 0, "No summary data"

        columns = [
            "iteration_al",
            "n_tasks",
            "n_systems",
            "n_validity_checks",
            "n_validity_success",
            "n_frames",
            "n_accuracy_checks",
            "n_accuracy_success",
            "n_train_checks",
            "n_train_success",
        ]
        available_columns = [column for column in columns if column in df.columns]
        df = df.loc[:, available_columns].sort_values("iteration_al")
        for column in available_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        fig, axes = plt.subplots(5, 1, figsize=figsize or (9, 13), sharex=True)
        frames_df = self._train_validation_frames_dataframe()

        plot_groups = [
            (
                axes[0],
                ("n_tasks", "n_systems"),
                "Tasks and Systems",
                "count",
            ),
            (
                axes[1],
                ("n_validity_checks", "n_validity_success"),
                "Validity",
                "checks",
            ),
            (
                axes[2],
                ("n_frames", "n_frames_training", "n_frames_validation"),
                "Frames Added",
                "frames",
            ),
            (
                axes[3],
                ("n_accuracy_checks", "n_accuracy_success"),
                "Accuracy Checks",
                "checks",
            ),
            (
                axes[4],
                ("n_train_checks", "n_train_success"),
                "Train Checks",
                "checks",
            ),
        ]
        markers = ("o", "s", "^", "D")
        labels = {
            "n_frames": "total",
            "n_frames_training": "training",
            "n_frames_validation": "validation",
        }

        for ax, group_columns, title, ylabel in plot_groups:
            plotted = False
            for marker, column in zip(markers, group_columns, strict=False):
                plot_df = frames_df if column in {"n_frames", "n_frames_training", "n_frames_validation"} else df
                x_values = plot_df["iteration_al"]
                y_column = "tot_frames" if column == "n_frames" and "tot_frames" in plot_df.columns else column
                if y_column not in plot_df.columns:
                    continue
                ax.plot(
                    x_values,
                    plot_df[y_column],
                    marker=marker,
                    markersize=6,
                    markeredgewidth=1.25,
                    markeredgecolor="white",
                    linewidth=2,
                    label=labels.get(column, column),
                )
                plotted = True
            assert plotted, f"No data available for {title}"
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(which="both", ls="--", linewidth=1)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelbottom=True)
            ax.legend()

        axes[-1].set_xlabel("iteration_al")
        fig.suptitle("Active Learning Summary")
        fig.tight_layout()

        g = type("G", (), {})()
        g.figure = fig
        return g

    def timings_table(self, figsize=None):
        _check_analysis_dependencies()
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(self.analysis.collect_table("timings", add_skip=False))
        assert len(df) > 0, "No timing data"

        phase_columns = [column for column in df.columns if column.endswith("[min]")]
        assert len(phase_columns) > 0, "No phase timing data"

        df = df.loc[:, ["iteration_al", *phase_columns]].sort_values("iteration_al")
        phase_df = df[phase_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        phase_df = phase_df.loc[:, phase_df.sum(axis=0) > 0]
        assert phase_df.shape[1] > 0, "No numeric phase timing data"

        x_positions = list(range(len(df)))
        fig_width = min(8, len(df) * 1.2)
        fig, ax = plt.subplots(figsize=figsize or (fig_width, 6))

        bottom = [0.0] * len(df)
        for column in phase_df.columns:
            values = phase_df[column].tolist()
            ax.bar(
                x_positions,
                values,
                bottom=bottom,
                label=column[: -len("[min]")],
            )
            bottom = [base + value for base, value in zip(bottom, values, strict=True)]

        ax.set_title("Active Learning Timings")
        ax.set_xlabel("iteration_al")
        ax.set_ylabel("time [min]")
        # ax.set_xticks(x_positions)
        # ax.set_xticklabels(df["iteration_al"].tolist())
        ax.grid(axis="y", which="both", ls="--", linewidth=1)
        ax.set_axisbelow(True)
        ax.legend(title="phase", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        g = type("G", (), {})()
        g.figure = fig
        return g

    def _set_seaorn(self, g, title: str):
        import seaborn as sns

        assert isinstance(g, sns.FacetGrid)

        sns.set_context("talk", font_scale=1)
        g.set_titles(size=20)
        g.add_legend(title_fontsize=18, bbox_to_anchor=(0.15, 1))
        g.figure.suptitle(title, fontsize=20)
        for ax in g.axes.flatten():
            ax.grid(which="both", ls="--", linewidth=1)
        g.figure.tight_layout(rect=(0, 0, 0.85, 0.95))
        return g

    def _check_histogram_dataframe(self, options: Literal["validity", "accuracy", "train_check"]):
        import pandas as pd

        df = pd.DataFrame(self.analysis.collect_table(options))
        assert len(df) > 0, f"No {options.replace('_', '-')} check data"
        assert "iteration_al" in df.columns, f"No iteration_al column in {options.replace('_', '-')} table"
        assert "success" in df.columns, f"No success column in {options.replace('_', '-')} table"

        if "task_id" not in df.columns:
            df = df.assign(task_id="all")

        columns = ["iteration_al", "task_id", "success"]
        if options in {"accuracy", "train_check"}:
            columns.extend(column for column in ("property", "metric") if column in df.columns)

        df = df.loc[:, columns].dropna(subset=["iteration_al", "task_id", "success"])
        assert len(df) > 0, f"No complete {options.replace('_', '-')} check rows"

        iteration_al = pd.to_numeric(df["iteration_al"], errors="coerce")
        df = df.assign(
            iteration_al=iteration_al if iteration_al.notna().all() else df["iteration_al"].astype(str),
            task_id=df["task_id"].astype(str),
            success=df["success"].astype(str),
        )
        if options in {"accuracy", "train_check"}:
            df = self._with_pairwise_failure_labels(df)

        return (
            df.groupby(["iteration_al", "task_id", "success"], observed=True, sort=False)
            .size()
            .reset_index(name="n_checks")
            .sort_values(["task_id", "iteration_al", "success"])
        )

    @staticmethod
    def _with_pairwise_failure_labels(df):
        df = df.copy()
        if "property" not in df.columns or "metric" not in df.columns:
            return df

        df = df.assign(
            property=df["property"].fillna("").astype(str),
            metric=df["metric"].fillna("").astype(str),
        )
        labels = df["success"].copy()
        failed = df["success"] != "OK"
        for _, group in df[failed].groupby(["iteration_al", "task_id"], sort=False):
            pairs = group.loc[:, ["property", "metric"]].drop_duplicates()
            properties = sorted(prop for prop in pairs["property"].unique() if prop)
            if len(pairs) > 1:
                label = "+".join(properties) if properties else "pairwise"
            else:
                pair = pairs.iloc[0]
                prop = pair["property"] or "pairwise"
                metric = pair["metric"] or "metric"
                label = f"{prop}_{metric}"
            labels.loc[group.index] = label
        return df.assign(success=labels)

    @staticmethod
    def _success_palette(success_values):
        failures = sorted(value for value in success_values if value != "OK")
        palette = {"OK": "tab:green"}
        failure_colors = (
            "tab:blue",
            "tab:orange",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
            "gold",
        )
        palette.update({failure: color for failure, color in zip(failures, cycle(failure_colors), strict=False)})
        return palette

    def check_histogram(
        self,
        options: Literal["validity", "accuracy", "train_check"],
        col_wrap: int = 3,
        height: float = 4,
        aspect: float = 1.25,
    ):
        _check_analysis_dependencies()
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        df = self._check_histogram_dataframe(options)
        hue_order = ["OK"] if (df["success"] == "OK").any() else []
        hue_order.extend(sorted(value for value in df["success"].unique() if value != "OK"))
        palette = self._success_palette(hue_order)

        task_ids = sorted(df["task_id"].unique())
        ncols = min(max(col_wrap, 1), len(task_ids))
        nrows = (len(task_ids) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(height * aspect * ncols + 2, height * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for ax, task_id in zip(axes_flat, task_ids, strict=False):
            task_df = df[df["task_id"] == task_id]
            pivot = task_df.pivot_table(
                index="iteration_al",
                columns="success",
                values="n_checks",
                aggfunc="sum",
                fill_value=0,
                sort=True,
            )
            for success in hue_order:
                if success not in pivot:
                    pivot[success] = 0
            pivot = pivot[hue_order]

            x_positions = range(len(pivot.index))
            bottoms = [0] * len(pivot.index)
            for success in hue_order:
                values = pivot[success].tolist()
                ax.bar(
                    x_positions,
                    values,
                    bottom=bottoms,
                    color=palette[success],
                    label=success,
                )
                bottoms = [base + value for base, value in zip(bottoms, values, strict=True)]

            ax.set_title(f"task_id={task_id}", fontsize=13)
            ax.set_xlabel("iteration_al")
            ax.set_ylabel("check count")
            ax.set_xticks(list(x_positions))
            ax.set_xticklabels([str(value) for value in pivot.index], rotation=90)
            ax.grid(axis="y", which="both", ls="--", linewidth=1)
            ax.set_axisbelow(True)

        for ax in axes_flat[len(task_ids) :]:
            ax.set_visible(False)

        handles = [Patch(facecolor=palette[value], label=value) for value in hue_order]
        fig.legend(
            handles=handles,
            title="check result",
            bbox_to_anchor=(0.86, 0.95),
            loc="upper left",
            borderaxespad=0,
        )
        fig.suptitle(f"{options.replace('_', ' ').title()} Checks by Task", fontsize=18)
        fig.tight_layout(rect=(0, 0, 0.84, 0.95))

        g = type("G", (), {})()
        g.figure = fig
        g.axes = axes
        return g

    def validity_check_histogram(self, col_wrap: int = 3, height: float = 4, aspect: float = 1.25):
        return self.check_histogram("validity", col_wrap=col_wrap, height=height, aspect=aspect)

    def accuracy_check_histogram(self, col_wrap: int = 3, height: float = 4, aspect: float = 1.25):
        return self.check_histogram("accuracy", col_wrap=col_wrap, height=height, aspect=aspect)

    def train_check_histogram(self, col_wrap: int = 3, height: float = 4, aspect: float = 1.25):
        return self.check_histogram("train_check", col_wrap=col_wrap, height=height, aspect=aspect)

    def validity_table(
        self,
        row: Literal["components-Z", "metric", "property"] | None = "property",
        col: Literal["components-Z", "metric", "property"] | None = "metric",
        hue: Literal["components-Z", "metric", "property"] | None = None,
        x: str = "iteration_al",
        include_hue_for_target: bool = True,
    ):
        _check_analysis_dependencies()
        import pandas as pd
        import seaborn as sns
        from matplotlib.lines import Line2D

        df = pd.DataFrame(self.analysis.collect_table("validity"))
        required_columns = {"value_type", "success"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Found empty values in validity table; "
                f"missing required column(s): {', '.join(sorted(missing_columns))}"
            )
        df = df[(df["value_type"] != "str") & (df["success"] != "OK")]
        assert len(df) > 0, "No failed validity data"
        y = "value"
        df = df.assign(
            **{
                "components-Z": lambda dff: dff["components"].astype(str) + " - " + dff["atom_type"].astype(str),
            }
        )
        markersize = 60
        g = sns.FacetGrid(
            df,
            row=row,
            col=col,
            hue=hue,
            margin_titles=True,
            height=7,
            aspect=1,
            sharey=False,
        )
        g.map(sns.scatterplot, x, y, marker="o", s=markersize).set(yscale="linear")
        g = self._set_seaorn(g, "Failed Validity Table")
        if include_hue_for_target:
            g.map(sns.scatterplot, x, "target", color="red", marker="D", s=markersize)
        else:
            row_values = g.row_names if row is not None else [None]
            col_values = g.col_names if col is not None else [None]
            for i, row_value in enumerate(row_values):
                for j, col_value in enumerate(col_values):
                    facet_df = df
                    if row is not None:
                        facet_df = facet_df[facet_df[row] == row_value]
                    if col is not None:
                        facet_df = facet_df[facet_df[col] == col_value]
                    if facet_df.empty:
                        continue
                    sns.scatterplot(
                        data=facet_df.drop_duplicates(subset=[x, "target"]).sort_values(x),
                        x=x,
                        y="target",
                        ax=g.axes[i, j],
                        color="red",
                        marker="D",
                        s=markersize,
                        legend=False,
                    )
        if g._legend is not None:
            g._legend.remove()

        legend_data = dict(g._legend_data)
        if not legend_data:
            legend_data["failed validity"] = Line2D([], [], color="C0", marker="o", linestyle="None", markersize=10)
        legend_data["target"] = Line2D(
            [],
            [],
            color="red",
            marker="D",
            linestyle="None",
            markersize=8,
        )
        g.add_legend(
            legend_data=legend_data,
            title=hue,
            title_fontsize=18,
            bbox_to_anchor=(0, 1),
            loc="upper left",
        )
        g.figure.tight_layout(rect=(0.12, 0, 1, 0.95))
        return g

    def summary_validity_table(
        self,
        groupby: Literal["task_id", "system_id"] = "task_id",
        x: str = "iteration_al",
        y: str = "value",
    ):
        _check_analysis_dependencies()
        import pandas as pd
        import seaborn as sns

        df = pd.DataFrame(self.analysis.collect_table("summary_validity", groupby=groupby))
        assert len(df) > 0, "No summary-validity data"
        assert groupby in df.columns, f"No {groupby} data in summary-validity table"

        value_vars = [
            column
            for column in ("n_tasks", "n_systems", "n_validity_checks", "n_validity_success", "n_frames")
            if column in df.columns
        ]
        assert len(value_vars) > 0, "No summary-validity values to plot"

        df = df.sort_values([groupby, x])
        for column in value_vars:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        id_vars = [column for column in (x, groupby) if column in df.columns]
        df_melt = df.melt(id_vars=id_vars, value_vars=value_vars).dropna(subset=[x, y])
        assert len(df_melt) > 0, "No numeric summary-validity data"
        df_melt = df_melt.assign(
            metric_group=lambda dff: dff["variable"].map(
                {
                    "n_tasks": "Tasks, Systems, and Frames",
                    "n_systems": "Tasks, Systems, and Frames",
                    "n_frames": "Tasks, Systems, and Frames",
                    "n_validity_checks": "Validity Checks",
                    "n_validity_success": "Validity Checks",
                }
            )
        ).dropna(subset=["metric_group"])

        g = sns.FacetGrid(
            df_melt,
            row="metric_group",
            col=groupby,
            hue="variable",
            margin_titles=False,
            height=4,
            aspect=1.4,
            sharey=False,
        )
        g.map(sns.lineplot, x, y, marker="o").set(yscale="linear")
        g = self._set_seaorn(g, f"Validity Summary by {groupby}")
        g.set_titles(row_template="", col_template="{col_name}", size=20)
        return g

    def accuracy_table(
        self,
        row: Literal["components-Z", "metric", "property"] | None = "property",
        col: Literal["components-Z", "metric", "property"] | None = "metric",
        hue: Literal["components-Z", "metric", "property"] | None = None,
        x: str = "iteration_al",
    ):
        _check_analysis_dependencies()
        import pandas as pd
        import seaborn as sns

        df = pd.DataFrame(self.analysis.collect_table("accuracy"))
        assert len(df) > 0, "No accuracy data"
        y = "value"
        df = df.assign(
            **{
                "components-Z": lambda dff: dff["components"].astype(str) + " - " + dff["atom_type"].astype(str),
            }
        )
        g = sns.FacetGrid(
            df,
            row=row,
            col=col,
            hue=hue,
            margin_titles=True,
            height=7,
            aspect=1,
            sharey=False,
        )
        g.map(sns.scatterplot, x, y, marker="o").set(yscale="linear")
        g = self._set_seaorn(g, "Accuracy Table")
        g.map(sns.lineplot, x, "target", color="red", linestyle="--", linewidth=2)
        return g

    def train_check_table(
        self,
        row: Literal["components-Z", "metric", "property"] | None = "property",
        col: Literal["components-Z", "metric", "property"] | None = "metric",
        hue: Literal["dataset", "components-Z", "metric", "property"] | None | str = "dataset",
        x: str = "iteration_al",
    ):
        _check_analysis_dependencies()
        import pandas as pd
        import seaborn as sns

        df = pd.DataFrame(self.analysis.collect_table("train_check"))
        assert len(df) > 0, "No train-check data"

        y = "value"
        df = df.assign(
            **{
                "components-Z": lambda dff: dff["components"].astype(str) + " - " + dff["atom_type"].astype(str),
                y: lambda dff: pd.to_numeric(dff[y], errors="coerce"),
                "target": lambda dff: pd.to_numeric(dff["target"], errors="coerce"),
            }
        ).dropna(subset=[x, y])
        assert len(df) > 0, "No numeric train-check data"

        if hue is not None and hue not in df.columns:
            hue = None

        g = sns.FacetGrid(
            df,
            row=row,
            col=col,
            hue=hue,
            margin_titles=True,
            height=7,
            aspect=1,
            sharey=False,
        )
        g.map(sns.scatterplot, x, y, marker="o").set(yscale="linear")
        g = self._set_seaorn(g, "Train Check Table")
        g.map(sns.lineplot, x, "target", color="red", linestyle="--", linewidth=2)
        return g

    def train_table(
        self,
        hue: Literal["dataset", "units", "property"] | None | str = "dataset",
        row: Literal["dataset", "units", "property"] | None | str = "property",
        col: Literal["dataset", "units", "property"] | None | str = "variable",
        x: str = "iteration_al",
        y: str = "value",
    ):
        _check_analysis_dependencies()
        import pandas as pd
        import seaborn as sns

        df: pd.DataFrame = pd.DataFrame(self.analysis.collect_table("train")).query("trained")
        assert len(df) > 0, "No Data Trained"

        base_excluded = {"iteration_al", "trained"}
        metadata_columns = {"type", "dataset", "property", "units", "run_directory"}
        candidate_value_vars = [
            "mae",
            "rmse",
            "r2",
            *[
                column
                for column in df.columns
                if column not in base_excluded | metadata_columns and pd.api.types.is_numeric_dtype(df[column])
            ],
        ]
        value_vars = [column for column in dict.fromkeys(candidate_value_vars) if column in df.columns]
        value_vars = [column for column in value_vars if df[column].notna().any()]
        assert len(value_vars) > 0, "No numeric training metrics available to plot"

        id_vars = [
            column
            for column in ("iteration_al", "property", "units", "dataset", "type", "run_directory")
            if column in df.columns
        ]
        df_melt = df.melt(value_vars=value_vars, id_vars=id_vars)

        if {"property", "units"}.issubset(df_melt.columns):
            df_melt = df_melt.assign(
                **{
                    "property": lambda dff: dff["property"].astype(str) + "|" + dff["units"].astype(str),
                }
            )

        facet_defaults = {"hue": hue, "row": row, "col": col}
        facet_fallbacks = {
            "hue": ("dataset", "type", None),
            "row": ("property", "dataset", None),
            "col": ("variable", "units", None),
        }
        resolved_facets = {}
        for facet_name, requested in facet_defaults.items():
            if requested is None or requested in df_melt.columns:
                resolved_facets[facet_name] = requested
                continue
            resolved = None
            for candidate in facet_fallbacks[facet_name]:
                if candidate is None or candidate in df_melt.columns:
                    resolved = candidate
                    break
            resolved_facets[facet_name] = resolved

        g = sns.FacetGrid(
            df_melt,
            row=resolved_facets["row"],
            col=resolved_facets["col"],
            hue=resolved_facets["hue"],
            margin_titles=True,
            height=7,
            aspect=1,
            sharey=False,
        )
        g.map(sns.lineplot, x, y, marker="o").set(yscale="linear")
        return self._set_seaorn(g, "Train Table")

    def dataset_property_distributions(
        self,
    ):
        dataset = self.analysis.loop.splitting.dataset
        fig, axes = PropertyDistributionPlot(title="Dataset Property Distributions").run(dataset)
        return fig
