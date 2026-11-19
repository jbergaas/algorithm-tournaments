"""Rebuild the Erdos02 network from the original Pajek file Erdos02.net.

The file declares named vertices in a *Vertices section and lists the edges as
adjacency lists in an *Edgeslist section: each line holds a vertex ID followed by
the IDs of its neighbours. For some vertices the list is spread over several
lines that each repeat the vertex ID. All lines of a vertex are joined.

Parsing is strict and stops with the line number on anything else:
  - *Vertices N comes first, followed by the lines 'ID "label"' for every
    vertex 1..N, each declared once;
  - after that only *Edgeslist sections and empty *Arcslist sections are
    accepted;
  - every ID on an adjacency line must be a declared vertex.

Writes, into a new output directory:
  reconstructed.mtx       Matrix Market "coordinate pattern symmetric". Vertex i is
                          Pajek vertex i, declared vertices without edges included.
                          One "row column" line with row > column per edge;
                          self-loops and repeated edges are removed.
  source_to_internal.tsv  Pajek ID, 0-based internal ID, matrix index and label of
                          every vertex.
  converter-record.json   counts of vertices, edges and adjacency lines, also
                          printed. "edges_last_line_only" is the number of edges
                          when only the last line of each vertex is kept (for
                          Erdos02 the 8,472 edges of the SuiteSparse copy).

Usage: python3 erdos_converter.py --input Erdos02.net --output-dir NEW_DIR
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")
# A vertex line: ID, whitespace, label in double quotes ("" inside the label is one ").
VERTEX_LINE = re.compile(r'([1-9][0-9]*)\s+"((?:[^"]|"")*)"')
MATRIX_COMMENT = "% Reconstructed from original Pajek Edgeslist; repeated source lines are appended.\n"


class PajekNetwork(NamedTuple):
    vertex_count: int
    labels: dict  # Pajek ID -> label
    adjacency_lines: list  # (vertex, [neighbours]) for every adjacency line, in file order


class PajekParser:
    """Reads a Pajek file one line at a time."""

    def __init__(self):
        self.vertex_count = None
        self.labels = {}
        self.adjacency_lines = []
        self.section = None
        self.has_edgeslist = False

    def read_line(self, line):
        if line.startswith("*"):
            self.start_section(line)
        elif self.section == "*vertices":
            self.add_vertex(line)
        elif self.section == "*edgeslist":
            self.add_adjacency_line(line)
        elif self.section == "*arcslist":
            raise ValueError("the *Arcslist section must be empty")
        else:
            raise ValueError("record before the first section")

    def start_section(self, line):
        fields = line.split()
        name = fields[0].lower()
        if name == "*vertices":
            if self.vertex_count is not None:
                raise ValueError("second *Vertices section")
            if len(fields) != 2 or not POSITIVE_INTEGER.fullmatch(fields[1]):
                raise ValueError("expected '*Vertices N' with N a positive integer")
            self.vertex_count = int(fields[1])
        else:
            if self.vertex_count is None:
                raise ValueError(f"{line!r} comes before *Vertices")
            if len(self.labels) != self.vertex_count:
                raise ValueError(f"only {len(self.labels)} of {self.vertex_count} vertices are declared")
            if len(fields) != 1 or name not in ("*arcslist", "*edgeslist"):
                raise ValueError(f"unsupported section {line!r}")
            if name == "*edgeslist":
                self.has_edgeslist = True
        self.section = name

    def add_vertex(self, line):
        match = VERTEX_LINE.fullmatch(line)
        if match is None:
            raise ValueError("expected a vertex ID and a quoted label")
        vertex = int(match[1])
        if vertex > self.vertex_count:
            raise ValueError(f"vertex ID {vertex} is larger than {self.vertex_count}")
        if vertex in self.labels:
            raise ValueError(f"vertex {vertex} is declared twice")
        self.labels[vertex] = match[2].replace('""', '"')

    def add_adjacency_line(self, line):
        fields = line.split()
        for field in fields:
            if not POSITIVE_INTEGER.fullmatch(field):
                raise ValueError(f"{field!r} is not a positive integer")
        ids = [int(field) for field in fields]
        for vertex in ids:
            if vertex > self.vertex_count:
                raise ValueError(f"vertex {vertex} is not declared")
        self.adjacency_lines.append((ids[0], ids[1:]))

    def finish(self):
        if self.vertex_count is None:
            raise ValueError("no *Vertices section")
        if len(self.labels) != self.vertex_count:
            raise ValueError(f"only {len(self.labels)} of {self.vertex_count} vertices are declared")
        if not self.has_edgeslist:
            raise ValueError("no *Edgeslist section")
        return PajekNetwork(self.vertex_count, self.labels, self.adjacency_lines)


def parse_pajek(text):
    """Return the PajekNetwork in the text; raise ValueError if it breaks the rules above."""
    parser = PajekParser()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        try:
            parser.read_line(line)
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from None
    return parser.finish()


def undirected_edges(adjacency_lines):
    """Return the set of (smaller ID, larger ID) pairs, self-loops included."""
    edges = set()
    for vertex, neighbours in adjacency_lines:
        for neighbour in neighbours:
            edges.add((min(vertex, neighbour), max(vertex, neighbour)))
    return edges


def without_loops(edges):
    return {edge for edge in edges if edge[0] != edge[1]}


def write_matrix(path, vertex_count, edges):
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        stream.write(MATRIX_COMMENT)
        stream.write(f"{vertex_count} {vertex_count} {len(edges)}\n")
        for smaller, larger in sorted(edges):
            stream.write(f"{larger} {smaller}\n")


def write_id_table(path, network):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["pajek_id_1based", "internal_id_0based", "matrix_index_1based", "label"])
        for pajek_id in range(1, network.vertex_count + 1):
            writer.writerow([pajek_id, pajek_id - 1, pajek_id, network.labels[pajek_id]])


def count_summary(network):
    all_edges = undirected_edges(network.adjacency_lines)
    edges = without_loops(all_edges)
    last_line_of_vertex = dict(network.adjacency_lines)  # later lines replace earlier ones
    lines_per_vertex = Counter(vertex for vertex, _ in network.adjacency_lines)
    return {
        "vertices": network.vertex_count,
        "edges": len(edges),
        "self_loops_removed": len(all_edges) - len(edges),
        "adjacency_lines": len(network.adjacency_lines),
        "vertices_with_several_lines": sum(1 for count in lines_per_vertex.values() if count > 1),
        "edges_last_line_only": len(without_loops(undirected_edges(last_line_of_vertex.items()))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="the Pajek file")
    parser.add_argument("--output-dir", required=True, help="output directory; must not exist yet")
    args = parser.parse_args()

    text = Path(args.input).read_bytes().decode("latin-1")
    try:
        network = parse_pajek(text)
    except ValueError as error:
        sys.exit(f"{args.input}: {error}")

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        sys.exit(f"{output_dir} already exists")
    edges = without_loops(undirected_edges(network.adjacency_lines))
    write_matrix(output_dir / "reconstructed.mtx", network.vertex_count, edges)
    write_id_table(output_dir / "source_to_internal.tsv", network)
    summary = json.dumps(count_summary(network), indent=2) + "\n"
    (output_dir / "converter-record.json").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
