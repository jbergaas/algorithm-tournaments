"""Reading and writing the CSV tables of the analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    """Writes rows with the given columns (default: every key, in order of first appearance).
    Lists and dicts are written as JSON, None as an empty field."""
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def read_csv(path: Path) -> Iterator[dict]:
    """The rows of a CSV file as dicts of strings, one at a time."""
    with open(path, newline='') as stream:
        yield from csv.DictReader(stream)
