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

STAGES = {
    "prepare": prepare.run,
    "features": features.run,
    "aggregate": aggregate.run,
    "figures": figures.run,
}


@click.command()
@click.option(
    "--stage",
    type=click.Choice(list(STAGES)),
    help="Run only this stage.",
)
@click.option(
    "--from",
    "start",
    type=click.Choice(list(STAGES)),
    help="Run this stage and every stage after it.",
)
def main(stage: str | None, start: str | None) -> None:
    matplotlib.use("Agg")  # write files, never open a window
    names = list(STAGES)
    if stage:
        selected = [stage]
    elif start:
        selected = names[names.index(start) :]
    else:
        selected = names
    paths = Paths()
    for name in selected:
        logger.info(f"--- stage: {name}")
        STAGES[name](paths)


if __name__ == "__main__":
    main()
