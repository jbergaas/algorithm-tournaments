"""Convert one SNAP edge-list download (.txt.gz) into Matrix Market inputs for the engine.

Outputs, in a new output directory:
  original.mtx    coordinate pattern general, one coordinate per SNAP data row, in file order.
                  Duplicate rows, reciprocal rows and self-loops are kept. Matrix index
                  i = 1 + rank of the SNAP node ID among all IDs that occur in any row.
  nodeid.mtx      array real general, n x 1: the SNAP node ID of every original matrix index
                  (the SuiteSparse *_nodeid.mtx convention).
  canonical.mtx   unique loop-free undirected edges, nodes without edges removed, IDs compacted
                  in increasing SNAP ID order. Coordinate pattern symmetric, one row > column entry
                  per edge, rows sorted.
  conversion.json counts and checksums.

Parsing is strict: '#' comment lines, otherwise exactly two non-negative decimal integers
separated by whitespace. CRLF line ends are accepted. Anything else fails.

Usage: python3 snap_to_mtx.py INPUT.txt.gz OUTPUT_DIRECTORY NAME SOURCE_URL
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROW = re.compile(rb"([0-9]+)[ \t]+([0-9]+)[ \t]*")
FILTER_COMMENT = "% Isolates removed after the engine's unique loop-free undirected symmetrization.\n"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def parse(raw):
    comments, arcs = [], []
    for number, line in enumerate(raw.split(b"\n"), start=1):
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            if number == raw.count(b"\n") + 1:  # text after the final newline is empty
                continue
            raise ValueError(f"empty line {number}")
        if line.startswith(b"#"):
            comments.append(line.decode("ascii"))
            continue
        match = ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed data line {number}: {line[:80]!r}")
        arcs.append((int(match.group(1)), int(match.group(2))))
    return comments, arcs


def write(path, text):
    with Path(path).open("x", encoding="ascii", newline="\n") as stream:
        stream.write(text)


def main():
    source, out, name, url = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    out.mkdir(parents=False, exist_ok=False)
    compressed = source.read_bytes()
    raw = gzip.decompress(compressed)
    comments, arcs = parse(raw)
    ids = sorted({v for arc in arcs for v in arc})
    rank = {v: i for i, v in enumerate(ids)}
    n0 = len(ids)
    source_sha = sha256_bytes(compressed)

    header = [f"% {name}: converted from SNAP edge list {source.name}",
              f"% source URL: {url}",
              f"% source SHA-256 (gzip file): {source_sha}",
              "% One coordinate per SNAP data row in file order; duplicate and reciprocal rows and",
              "% self-loops are kept. Matrix index = 1 + rank of the SNAP node ID among all IDs that",
              "% occur in a row; nodeid.mtx lists the SNAP node ID of every index.",
              *[f"% SNAP: {c}" for c in comments]]
    lines = ["%%MatrixMarket matrix coordinate pattern general", *header, f"{n0} {n0} {len(arcs)}"]
    lines += [f"{rank[u] + 1} {rank[v] + 1}" for u, v in arcs]
    write(out / "original.mtx", "\n".join(lines) + "\n")

    idlines = ["%%MatrixMarket matrix array real general",
               f"% {name}: SNAP node ID of every original.mtx matrix index (1-based, in order)",
               f"{n0} 1", *[str(v) for v in ids]]
    write(out / "nodeid.mtx", "\n".join(idlines) + "\n")

    loops = sum(1 for u, v in arcs if u == v)
    undirected = {(max(u, v), min(u, v)) for u, v in arcs if u != v}
    distinct_arcs = len(set(arcs))
    nonloop_arcs = {(u, v) for u, v in arcs if u != v}
    reciprocal_pairs = sum(1 for u, v in nonloop_arcs if u < v and (v, u) in nonloop_arcs)
    positive = sorted({v for e in undirected for v in e})
    loop_only = sorted(set(ids) - set(positive))
    new = {v: i for i, v in enumerate(positive)}
    neighbours = {}
    for a, b in undirected:
        neighbours.setdefault(new[a], []).append(new[b])
    n, m = len(positive), len(undirected)
    rows = [f"{r + 1} {c + 1}" for r in range(n) for c in sorted(neighbours.get(r, []))]
    write(out / "canonical.mtx", "%%MatrixMarket matrix coordinate pattern symmetric\n"
          + FILTER_COMMENT + f"{n} {n} {m}\n" + "\n".join(rows) + "\n")

    record = {
        "schema": "snap-edge-list-conversion/v1",
        "dataset": name,
        "source_url": url,
        "input": {"path": str(source), "sha256": source_sha, "bytes": len(compressed),
                  "decompressed_sha256": sha256_bytes(raw), "decompressed_bytes": len(raw)},
        "snap_comment_lines": comments,
        "data_rows": len(arcs),
        "distinct_rows": distinct_arcs,
        "duplicate_rows": len(arcs) - distinct_arcs,
        "self_loop_rows": loops,
        "reciprocal_unordered_pairs": reciprocal_pairs,
        "distinct_node_ids": n0,
        "min_node_id": ids[0],
        "max_node_id": ids[-1],
        "unique_undirected_loop_free_edges": m,
        "positive_degree_nodes": n,
        "loop_only_node_ids": loop_only,
        "outputs": {f: {"sha256": sha256_file(out / f), "bytes": (out / f).stat().st_size}
                    for f in ("original.mtx", "nodeid.mtx", "canonical.mtx")},
        "converter": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "conventions": {
            "original_matrix_index": "1 + rank of SNAP node ID among all IDs occurring in a row",
            "canonical_internal_id": "0-based rank of SNAP node ID among positive-degree IDs",
        },
    }
    write(out / "conversion.json", json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("snap_comment_lines", "outputs")}, indent=1))


if __name__ == "__main__":
    main()
