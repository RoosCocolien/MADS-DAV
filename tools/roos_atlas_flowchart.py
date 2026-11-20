#!/usr/bin/env python3
"""Redraw "The flowchart as atlas reads it" in `roos_pipeline/README.md` from atlas's graph.

atlas has no Mermaid export, so this asks it for the graph as JSON (`atlas analyze`) and
draws the chart from that: one box per pipeline module reached from `__main__.main`, one
cylinder per file atlas sees that module read or write, and the stages numbered in the
order `main` calls them. The chart lands between the two `atlas-flowchart` markers in the
README; everything around them stays mine. Lefthook runs it when `roos_pipeline/` changes.

    uv run python tools/roos_atlas_flowchart.py           # rewrite the README
    uv run python tools/roos_atlas_flowchart.py --check   # only say whether it is stale
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess  # noqa: S404 — runs atlas, a fixed local command
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "roos_pipeline" / "README.md"
PACKAGE = "roos_pipeline"
ENTRY = f"{PACKAGE}.__main__.main"
START = "<!-- atlas-flowchart:start (drawn by tools/roos_atlas_flowchart.py, do not edit) -->"
END = "<!-- atlas-flowchart:end -->"
# Since 0.74 atlas installs as `atlas-code`, in its own folder that is not on PATH.
ATLAS_NAMES = ("atlas", "atlas-code")
ATLAS_DEFAULT = (
    Path.home() / "AppData" / "Local" / "Programs" / "atlas-code" / "atlas-code.exe"
)


# Since 0.74 atlas also follows the tests' calls, whose files sit under pytest's
# `tmp_path`; those are not places of the pipeline itself.
TEST_PLACE = "{tmp_path}"


def find_atlas() -> str:
    """Path to the atlas binary: `atlas` or `atlas-code` on PATH, else the default install."""
    for name in ATLAS_NAMES:
        found = shutil.which(name)
        if found:
            return found
    if ATLAS_DEFAULT.is_file():
        return str(ATLAS_DEFAULT)
    raise SystemExit(
        f"atlas not found (tried {', '.join(ATLAS_NAMES)} on PATH and {ATLAS_DEFAULT}); "
        "install it or skip with LEFTHOOK_EXCLUDE"
    )


def read_graph() -> dict:
    """Run `atlas analyze` on the whole repository and return its graph.

    The whole repository, not `--package roos_pipeline`: on its own the package loses
    the calls into `wa_analyzer` and `goad_toolkit`, and with them the chat export.
    """
    atlas = find_atlas()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        subprocess.run(  # noqa: S603 — fixed arguments, no shell
            [atlas, "analyze", str(ROOT), "--out", str(out)],
            check=True,
            capture_output=True,
        )
        return json.loads(out.read_text(encoding="utf-8"))


def module_of(node: dict) -> str:
    return node.get("module") or node["id"]


def pipeline_modules(graph: dict) -> tuple[list[str], dict[str, set[str]]]:
    """The package's modules `main` reaches, and which of them uses which.

    A module is reached when one of its functions is called or one of its classes
    instantiated, starting from `main`; calls that leave the package are followed too,
    so a stage that reads through `wa_analyzer` still counts as reading.
    """
    nodes = {n["id"]: n for n in graph["nodes"]}
    out_edges = defaultdict(list)
    for e in graph["edges"]:
        if e["kind"] in ("calls", "instantiates") and e["dst"] in nodes:
            out_edges[e["src"]].append(e["dst"])
    # a class's methods run when it is instantiated
    for e in graph["edges"]:
        if e["kind"] == "contains" and nodes.get(e["src"], {}).get("kind") == "class":
            out_edges[e["src"]].append(e["dst"])

    seen, stack = set(), [ENTRY]
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(out_edges[current])

    in_package = {
        m for m in map(module_of, (nodes[i] for i in seen)) if m.startswith(PACKAGE)
    }
    uses = defaultdict(set)
    for src in seen:
        for dst in out_edges[src]:
            a, b = module_of(nodes[src]), module_of(nodes[dst])
            if a != b and a in in_package and b in in_package:
                uses[a].add(b)
    return sorted(in_package), uses


def stage_order(graph: dict) -> list[str]:
    """The stage modules in the order `__main__` calls their `run`, by line."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    main_module = module_of(nodes[ENTRY])
    calls = []
    for n in graph["nodes"]:
        if module_of(n) != main_module:
            continue
        for site in n.get("sites", []):
            target = nodes.get(site["to"])
            if target and module_of(target) != main_module and target["name"] == "run":
                calls.append((n["span"][0], site["line"], module_of(target)))
    return [module for *_, module in sorted(calls)]


def touches(graph: dict, modules: list[str]) -> dict[str, dict[str, set[str]]]:
    """Per module, the files it reads and writes, as atlas names them.

    What a function outside the package touches is counted for the package module
    that called it. A module that writes into a folder atlas also names as a whole
    (`figures/pipeline/`) gets the folder, not each file in it.
    """
    nodes = {n["id"]: n for n in graph["nodes"]}
    callees = defaultdict(list)
    for e in graph["edges"]:
        if e["kind"] in ("calls", "instantiates") and e["dst"] in nodes:
            callees[e["src"]].append(e["dst"])

    found = {m: {"disk_read": set(), "disk_write": set()} for m in modules}
    for n in graph["nodes"]:
        owner = module_of(n)
        if owner not in found:
            continue
        stack, seen = [n["id"]], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for t in nodes[current].get("touches", []):
                if t["kind"] in found[owner] and t.get("artifact", "").startswith(
                    TEST_PLACE
                ):
                    continue
                if t["kind"] in found[owner] and t.get("artifact"):
                    found[owner][t["kind"]].add(t["artifact"])
            stack.extend(
                c
                for c in callees[current]
                if not module_of(nodes[c]).startswith(PACKAGE)
            )

    for kinds in found.values():
        for kind, artifacts in kinds.items():
            folders = {a for a in artifacts if a.endswith("/")}
            kinds[kind] = {
                next((f for f in folders if a.startswith(f) and a != f), a)
                for a in artifacts
            }
    return found


def place_id(artifact: str) -> str:
    return "p_" + re.sub(r"\W+", "_", place_label(artifact)).strip("_")


def place_label(artifact: str) -> str:
    if artifact.endswith("/"):
        return artifact
    folder, _, name = artifact.rpartition("/")
    return name if folder.startswith("data/processed/roos") else artifact


def mermaid(graph: dict) -> str:
    nodes = {n["id"]: n for n in graph["nodes"]}
    modules, uses = pipeline_modules(graph)
    stages = stage_order(graph)
    places = touches(graph, modules)
    main_module = module_of(nodes[ENTRY])
    options = " / ".join(nodes[ENTRY].get("entry", {}).get("options", []))

    lines = ["flowchart LR", f'    main["__main__.main<br/>{options}"]']
    # stages first, in the order they run, then the modules they lean on
    boxes = stages + [m for m in modules if m != main_module and m not in stages]
    for m in boxes:
        lines.append(f'    {m.rsplit(".", 1)[-1]}["{m.rsplit(".", 1)[-1]}"]')

    all_places = sorted(
        {a for kinds in places.values() for arts in kinds.values() for a in arts}
    )
    for a in all_places:
        lines.append(f'    {place_id(a)}[("{place_label(a)}")]')
    lines.append("")

    for i, m in enumerate(stages, start=1):
        lines.append(f'    main -- "{i}" --> {m.rsplit(".", 1)[-1]}')
    for m in boxes:
        short = m.rsplit(".", 1)[-1]
        for a in sorted(places[m]["disk_read"]):
            lines.append(f"    {place_id(a)} --> {short}")
        for a in sorted(places[m]["disk_write"]):
            lines.append(f"    {short} --> {place_id(a)}")
    for m in boxes:
        if m in stages:
            continue
        for user in sorted(u for u, used in uses.items() if m in used):
            source = "main" if user == main_module else user.rsplit(".", 1)[-1]
            lines.append(f'    {source} -. "calls, no data" .-> {m.rsplit(".", 1)[-1]}')

    lines += [
        "",
        "    classDef code fill:#fde3cf,stroke:#D55E00,color:#000",
        "    classDef file fill:#fff,stroke:#0072B2,color:#000",
        "    class "
        + ",".join(["main", *(m.rsplit(".", 1)[-1] for m in boxes)])
        + " code",
        "    class " + ",".join(place_id(a) for a in all_places) + " file",
    ]
    return "\n".join(lines)


def with_chart(readme: str, chart: str) -> str:
    """Put `chart` between the markers, in the README's own line endings."""
    newline = "\r\n" if "\r\n" in readme else "\n"
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        raise SystemExit(f"{README} has no {START} … {END} block to fill")
    block = newline.join([START, "```mermaid", *chart.split("\n"), "```", END])
    return pattern.sub(lambda _: block, readme)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the README is stale"
    )
    args = parser.parse_args()

    readme = README.read_bytes().decode("utf-8")
    updated = with_chart(readme, mermaid(read_graph()))
    if updated == readme:
        return 0
    if args.check:
        print(
            f"{README.relative_to(ROOT)}: the atlas flowchart is stale; run without --check"
        )
        return 1
    README.write_bytes(updated.encode("utf-8"))
    print(f"redrew the atlas flowchart in {README.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
