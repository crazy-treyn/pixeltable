"""
Full-text search (FTS) demo: create a table, add a GIN FTS index, filter and rank with `col.search()`.

Use your existing Python environment (the one where you develop Pixeltable). Install marimo into it if needed:

  pip install marimo
  # or: uv pip install marimo

Run from the repository root with `pixeltable` importable (editable install is typical):

  pip install -e .
  python tool/marimo_fts_demo.py

Interactive UI:

  marimo run tool/marimo_fts_demo.py
  marimo edit tool/marimo_fts_demo.py

If you use uv with this repo, prefer the default project env without dev groups (dev pulls optional ML stacks
that can fail to build, e.g. onnxsim), and add marimo only:

  uv run --no-dev --with marimo marimo edit tool/marimo_fts_demo.py

Avoid `uv run --with-editable .` without `--no-dev`: that can still pull dev extras.

If you use an external DB (`PIXELTABLE_DB_CONNECT_STR`), unset it or use a dedicated environment;
this notebook configures the same variables as `tests/conftest.py` for local embedded Postgres.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import tempfile
    import uuid
    from pathlib import Path

    import pixeltable as pxt
    from pixeltable.config import Config
    from pixeltable.env import Env

    return Config, Env, Path, mo, os, pxt, tempfile, uuid


@app.cell
def _(Config, Env, Path, mo, os, pxt, tempfile, uuid):
    # Configure env and initialize Pixeltable (embedded Postgres).
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    shared_home = Path.home() / ".pixeltable"
    shared_home.mkdir(parents=True, exist_ok=True)

    demo_base = Path(tempfile.mkdtemp(prefix="pxt_marimo_fts_"))
    home_dir = demo_base / ".pixeltable"
    home_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PIXELTABLE_HOME"] = str(home_dir)
    os.environ["PIXELTABLE_CONFIG"] = str(shared_home / "config.toml")
    os.environ["PIXELTABLE_DB"] = f"fts_marimo_{uuid.uuid4().hex[:16]}"
    os.environ["PIXELTABLE_PGDATA"] = str(shared_home / "pgdata")
    os.environ["PIXELTABLE_START_DASHBOARD"] = "false"
    os.environ.setdefault("PIXELTABLE_API_URL", "https://preprod-internal-api.pixeltable.com")
    os.environ["FIFTYONE_DATABASE_DIR"] = str(home_dir / ".fiftyone")

    reinit_db = os.environ.get("PIXELTABLE_DB_CONNECT_STR") is None
    Env._init_env(reinit_db=reinit_db)
    pxt.init()
    Config.init({}, reinit=True)

    fts_available = not Env.get().is_using_cockroachdb
    env_banner = mo.md(
        f"**PIXELTABLE_HOME:** `{home_dir}`  \n"
        f"**PIXELTABLE_DB:** `{os.environ['PIXELTABLE_DB']}`  \n"
        f"**FTS available:** `{fts_available}` (not on CockroachDB)  \n"
        f"**reinit_db:** `{reinit_db}`"
    )
    return env_banner, fts_available


@app.cell
def _(fts_available, mo, pxt):
    # Sample rows from the FTS plan: filter, stemming, websearch OR, and ranking.
    if fts_available:
        rows = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaped over the sleepy dog.",
            "Pixeltable is a declarative AI data infrastructure.",
            "The cat in the hat.",
            "PostgreSQL full text search is powerful and fast.",
            "The quick brown fox.",
        ]
        table = pxt.create_table("marimo_fts_demo", {"text": pxt.String}, if_exists="replace")
        table.insert([{"text": s} for s in rows])
        table.add_fts_index("text")

        fox_rows = table.where(table.text.search("fox")).select(table.text).collect()
        jump_rows = table.where(table.text.search("jumping")).select(table.text).collect()

        or_rows = table.where(table.text.search("quick OR fast")).select(table.text).collect()

        q = table.text.search("quick brown fox")
        ranked = table.select(table.text, rank=q.rank).where(q).order_by(q.rank, asc=False).collect()

        summary = mo.md(
            "**Filter `fox`:** "
            f"{len(fox_rows)} rows  \n"
            "**Stem `jumping` → jump:** "
            f"{len(jump_rows)} rows  \n"
            "**`quick OR fast`:** "
            f"{len(or_rows)} rows  \n"
            "**Rank `quick brown fox`:** "
            f"{len(ranked)} rows"
        )
        jump_out, or_out, ranked_out, summary_out = jump_rows, or_rows, ranked, summary
    else:
        jump_out, or_out, ranked_out, summary_out = (
            None,
            None,
            None,
            mo.md("**Full-text search** requires PostgreSQL. Skipping demo on CockroachDB."),
        )
    return jump_out, or_out, ranked_out, summary_out


@app.cell
def _(fts_available, mo, pxt):
    # French language index (same pattern as test_french_language_index).
    if fts_available:
        fr = pxt.create_table("marimo_fts_fr", {"text": pxt.String}, if_exists="replace")
        fr.insert([{"text": "Les chats sont dans le jardin."}])
        fr.add_fts_index("text", language="french")
        fr_hits = fr.where(fr.text.search("chat")).select(fr.text).collect()
        fr_md = mo.md(f"**French index, query `chat`:** {len(fr_hits)} row(s)")
    else:
        fr_md = mo.md("")
    return (fr_md,)


@app.cell
def _(env_banner, fts_available, fr_md, jump_rows, mo, or_rows, ranked, summary):
    # Show structured results (marimo renders the last expression).
    import pandas as pd

    if fts_available:
        rank_df = pd.DataFrame(ranked) if ranked else pd.DataFrame()
        jump_df = pd.DataFrame(jump_rows) if jump_rows else pd.DataFrame()
        or_df = pd.DataFrame(or_rows) if or_rows else pd.DataFrame()

        result = mo.vstack(
            [
                env_banner,
                summary,
                mo.md("### Ranked matches (`quick brown fox`)"),
                mo.ui.table(rank_df),
                mo.md("### Stemming (`jumping`)"),
                mo.ui.table(jump_df),
                mo.md("### Websearch OR (`quick OR fast`)"),
                mo.ui.table(or_df),
                fr_md,
            ]
        )
    else:
        result = mo.vstack([env_banner, mo.md("FTS demo skipped."), summary])
    result
    return


if __name__ == "__main__":
    app.run()
