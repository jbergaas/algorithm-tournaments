# Networks

The 56 networks of the study, 11 real and 45 generated, one Matrix Market file per network
(`coordinate pattern symmetric`, nodes numbered from 1). `configs/full-study.json` uses them as
they are.

Every file is a simple undirected graph without isolated nodes. Self-loops and duplicate edges
were removed, directed networks were made undirected, and nodes without edges were removed with
`build/release/isolate_filter --input IN.mtx --output-dir NEW_DIR`, which keeps the order of the
other nodes. The one exception is `ca-CondMat.mtx`, which keeps the provider's 58 self-loops; the
engine ignores them.

## Real networks

| Network | Nodes | Edges | Source | Changes |
|---|---:|---:|---|---|
| ego-Facebook | 4,039 | 88,234 | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) | converted from the edge list |
| ca-GrQc | 5,241 | 14,484 | [SuiteSparse](https://sparse.tamu.edu/SNAP/ca-GrQc) | self-loops and 1 node removed |
| Erdos02 | 6,927 | 11,850 | [Pajek](https://web.archive.org/web/20060813124748id_/http://vlado.fmf.uni-lj.si/pub/networks/data/Erdos/Erdos02.net) | rebuilt from the original file (below) |
| wiki-Vote | 7,115 | 100,762 | [SuiteSparse](https://sparse.tamu.edu/SNAP/wiki-Vote) | made undirected; 1,182 nodes without edges removed |
| ca-HepTh | 9,875 | 25,973 | [SNAP](https://snap.stanford.edu/data/ca-HepTh.html) | converted from the edge list; self-loops and 2 nodes removed |
| ca-HepPh | 12,006 | 118,489 | [SuiteSparse](https://sparse.tamu.edu/SNAP/ca-HepPh) | self-loops and 2 nodes removed |
| ca-CondMat | 23,133 | 93,439 | [SuiteSparse](https://sparse.tamu.edu/SNAP/ca-CondMat) | none |
| email-Enron | 36,692 | 183,831 | [SuiteSparse](https://sparse.tamu.edu/SNAP/email-Enron) | none |
| soc-Epinions1 | 75,879 | 405,740 | [SNAP](https://snap.stanford.edu/data/soc-Epinions1.html) | converted from the edge list; made undirected |
| com-DBLP | 317,080 | 1,049,866 | [SuiteSparse](https://sparse.tamu.edu/SNAP/com-DBLP) | none (SNAP publishes only the largest connected component) |
| com-Youtube | 1,134,890 | 2,987,624 | [SuiteSparse](https://sparse.tamu.edu/SNAP/com-Youtube) | none (SNAP publishes only the largest connected component) |

The SuiteSparse networks are SNAP networks as republished in the SuiteSparse Matrix Collection.
The three SNAP networks were downloaded on 4 November 2026 and converted with
`study-scripts/input-prep/snap_to_mtx.py`. The converter's files for ego-Facebook and
soc-Epinions1 differ from the files here in the comment on their second line. The engine hashes
the whole file into Random's seeds, so use the files here to repeat Random's draws exactly.

The SuiteSparse copy of Erdos02 has only 8,472 of the 11,850 edges of the original Pajek file, in
which the edge lists of 54 nodes span several lines. `Erdos02.mtx` was therefore rebuilt from the
original file, kept by the Internet Archive (link in the table), with
`study-scripts/input-prep/erdos_converter.py`.

## Generated networks

There are 13 settings with three instances each (`-i0`, `-i1`, `-i2`) at about 10,000 nodes, and
two settings repeated at about 50,000 nodes (`er-d8-n50000`, `cl-g2.5-n50000`).

- `er-d4` to `er-d32`: Erdős–Rényi random graphs with mean degree 4, 8, 16 or 32.
- `cl-g2.2`, `cl-g2.5`, `cl-g3.0`: Chung–Lu graphs with mean degree 8 and power-law exponent 2.2,
  2.5 or 3.0.
- `ws-p0.01`, `ws-p0.1`, `ws-p1.0`: Watts–Strogatz graphs with 8 neighbours per node and
  rewiring probability 0.01, 0.1 or 1.0.
- `collab-s10`, `collab-s50`, `collab-s200`: coauthorship graphs in which each paper joins its
  authors. Small papers (2 to 4 authors) are added until the mean degree is 6, then papers with
  10, 50 or 200 authors until it reaches 8.

The second line of each file gives the model, its parameters and the random seed. Nodes without
edges were removed, so some graphs have slightly fewer than 10,000 or 50,000 nodes.
`study-scripts/input-prep/generate_networks.py` makes identical files with the package versions in
`requirements.txt`.

## Licences and citation

The MIT licence of the repository covers the code, not the networks:

- **SuiteSparse** (ca-GrQc, wiki-Vote, ca-HepPh, ca-CondMat, email-Enron, com-DBLP, com-Youtube):
  CC BY 4.0. ca-GrQc, wiki-Vote and ca-HepPh are modified as listed in the table. Cite [8] and [9].
- **SNAP** (ego-Facebook, ca-HepTh, soc-Epinions1): no licence stated. SNAP asks for a citation
  of the collection [10] and of each dataset.
- **Pajek** (Erdos02): CC BY-NC-SA 2.5, the licence of the Pajek datasets. `Erdos02.mtx` is an
  adaptation of the original file and is shared under the same licence. Cite [11].
- **Generated networks** (the 45 `er-`, `cl-`, `ws-` and `collab-` files) and the ten-node test
  graph `tests/data/test.mtx`: made for this study and under the MIT licence of the code.

The provider files in `tests/data/` (ca-GrQc, ca-HepPh, ca-CondMat, email-Enron, wiki-Vote and
Erdos02) are under the same licences as above.

Datasets:

- [1] J. Leskovec, J. Kleinberg and C. Faloutsos. Graph Evolution: Densification and Shrinking
  Diameters. ACM TKDD 1(1), 2007. (ca-GrQc, ca-HepTh, ca-HepPh, ca-CondMat)
- [2] J. Leskovec, K. Lang, A. Dasgupta and M. Mahoney. Community Structure in Large Networks:
  Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics
  6(1), 29–123, 2009; and B. Klimt and Y. Yang. Introducing the Enron Corpus. CEAS, 2004.
  (email-Enron)
- [3] J. Leskovec, D. Huttenlocher and J. Kleinberg. Signed Networks in Social Media. CHI, 2010.
  (wiki-Vote)
- [4] J. Leskovec, D. Huttenlocher and J. Kleinberg. Predicting Positive and Negative Links in
  Online Social Networks. WWW, 2010. (wiki-Vote)
- [5] J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.
  (ego-Facebook)
- [6] M. Richardson, R. Agrawal and P. Domingos. Trust Management for the Semantic Web. ISWC,
  2003. (soc-Epinions1)
- [7] J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth.
  ICDM, 2012. (com-DBLP, com-Youtube)

Collections:

- [8] T. A. Davis and Y. Hu. The University of Florida Sparse Matrix Collection. ACM Transactions
  on Mathematical Software 38(1), 2011.
- [9] S. Kolodziej et al. The SuiteSparse Matrix Collection Website Interface. Journal of Open
  Source Software 4(35), 1244, 2019.
- [10] J. Leskovec and A. Krevl. SNAP Datasets: Stanford Large Network Dataset Collection.
  https://snap.stanford.edu/data, 2014.
- [11] V. Batagelj and A. Mrvar. Pajek datasets. http://vlado.fmf.uni-lj.si/pub/networks/data/,
  2006. SuiteSparse lists J. Grossman, P. Iain and R. Castro (the Erdős Number Project) as the
  authors of the network.
