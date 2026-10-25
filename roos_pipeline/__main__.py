"""Run the pipeline from the command line.

    uv run python -m roos_pipeline                    # all four stages
    uv run python -m roos_pipeline --stage figures    # only redraw, from stage-3 files
    uv run python -m roos_pipeline --from aggregate   # aggregate, then figures

Each stage reads the previous stage's file from data/processed/roos/, so any stage can
run on its own as long as the stage before it has run once.
"""

from __future__ import annotations

import click
import matplotlib
from loguru import logger

from roos_pipeline import aggregate, features, figures, prepare
from roos_pipeline.settings import Paths

# The order of the stages, for --stage/--from. `run_stages` calls them in this same
# order, written out so the calls can be read (and drawn by atlas) one by one;
# test_main_calls_the_stages_in_the_order_of_STAGES keeps the two in step.
STAGES = ("prepare", "features", "aggregate", "figures")


def selected_stages(stage: str | None, start: str | None) -> list[str]:
    """--stage: that one stage; --from: that stage and every one after it; else all."""
    if stage:
        return [stage]
    if start:
        return list(STAGES[STAGES.index(start) :])
    return list(STAGES)


def run_stages(paths: Paths, selected: list[str]) -> None:
    """Run the selected stages, always in pipeline order."""
    if "prepare" in selected:
        logger.info("--- stage: prepare")
        prepare.run(paths)
    if "features" in selected:
        logger.info("--- stage: features")
        features.run(paths)
    if "aggregate" in selected:
        logger.info("--- stage: aggregate")
        aggregate.run(paths)
    if "figures" in selected:
        logger.info("--- stage: figures")
        figures.run(paths)


@click.command()
@click.option(
    "--stage",
    type=click.Choice(STAGES),
    help="Run only this stage.",
)
@click.option(
    "--from",
    "start",
    type=click.Choice(STAGES),
    help="Run this stage and every stage after it.",
)
def main(stage: str | None, start: str | None) -> None:
    matplotlib.use("Agg")  # write files, never open a window
    run_stages(Paths(), selected_stages(stage, start))


if __name__ == "__main__":
    main()
