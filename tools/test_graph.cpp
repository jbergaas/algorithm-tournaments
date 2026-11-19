// The undirected view (graph.hpp): every entry links its two nodes once, whatever its direction,
// repetition or self-loop, for graphs built in memory and for Matrix Market files alike; and the
// algorithms and the spread give the same results on equivalent encodings of one graph.

#include "check.hpp"
#include "engine/algorithms.hpp"
#include "engine/graph.hpp"
#include "engine/spread.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::Graph;
using engine::Owner;
using Entries = std::vector<std::pair<uint32_t, uint32_t>>;
using Relation = std::vector<std::vector<uint32_t>>;

const std::vector<std::string> kRankedAlgorithms = {"degree", "coverage_greedy", "pagerank", "betweenness"};

Graph makeGraph(uint32_t nodeCount, const Entries& entries) {
    Graph::Builder builder(nodeCount);
    for (const auto& [from, to] : entries) builder.addEdge(from, to);
    return builder.build();
}

Relation neighborLists(const Graph& graph) {
    Relation lists;
    for (uint32_t node = 0; node < graph.nodeCount(); ++node) {
        const auto neighbors = graph.neighbors(node);
        lists.emplace_back(neighbors.begin(), neighbors.end());
    }
    return lists;
}

// The unique, loop-free undirected relation of the entries, built without Graph.
Relation symmetrize(uint32_t nodeCount, const Entries& entries) {
    Relation relation(nodeCount);
    for (const auto& [from, to] : entries) {
        if (from == to) continue;
        relation[from].push_back(to);
        relation[to].push_back(from);
    }
    for (auto& neighbors : relation) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    return relation;
}

// A graph that lists every edge of `relation` both ways, so its entries differ from the input's.
Graph bothWays(const Relation& relation) {
    Graph::Builder builder(static_cast<uint32_t>(relation.size()));
    for (uint32_t node = 0; node < relation.size(); ++node) {
        for (uint32_t neighbor : relation[node]) builder.addEdge(node, neighbor);
    }
    return builder.build();
}

// The 0-based entries of a Matrix Market file, read without the engine's loader. A symmetric file
// lists each off-diagonal entry once; it stands for both directions.
Entries rawEntries(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    const bool symmetric = line.find("symmetric") != std::string::npos;
    while (std::getline(file, line) && line.starts_with('%')) {
    }
    Entries entries;
    uint32_t row = 0;
    uint32_t column = 0;
    while (file >> row >> column) {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        entries.emplace_back(row - 1, column - 1);
        if (symmetric && row != column) entries.emplace_back(column - 1, row - 1);
    }
    return entries;
}

std::vector<uint32_t> candidates(const Graph& graph, const std::string& algorithm, uint32_t count) {
    if (algorithm == "degree") return engine::degreeCandidates(graph, count);
    if (algorithm == "coverage_greedy") return engine::coverageGreedyCandidates(graph, count);
    if (algorithm == "pagerank") return engine::pageRankCandidates(graph, count);
    auto ranking = engine::rankByScore(engine::betweennessScores(graph, 4));
    ranking.resize(std::min<size_t>(count, ranking.size()));
    return ranking;
}

// Both graphs must give the same Degree, Coverage Greedy and PageRank candidates and the same
// PageRank and Betweenness scores (equal scores give equal rankings).
void checkEquivalent(const Graph& first, const Graph& second, uint32_t count, const std::string& label) {
    for (const std::string algorithm : {"degree", "coverage_greedy", "pagerank"}) {
        checkEqual(candidates(first, algorithm, count), candidates(second, algorithm, count), label + ": " + algorithm);
    }
    checkEqual(engine::pageRankScores(first), engine::pageRankScores(second), label + ": PageRank scores");
    checkEqual(engine::betweennessScores(first, 4), engine::betweennessScores(second, 4),
               label + ": Betweenness scores");
}

void builder() {
    const auto chain = makeGraph(3, {{0, 1}, {1, 2}});
    check(chain.nodeCount() == 3 && chain.entryCount() == 2 && chain.edgeCount() == 2, "directed chain counts");
    check(chain.degree(0) == 1 && chain.degree(1) == 2 && chain.degree(2) == 1, "each entry links both of its nodes");
    checkEqual(neighborLists(chain), {{1}, {0, 2}, {1}}, "directed chain neighbours");

    const auto mixed = makeGraph(3, {{0, 0}, {0, 1}, {0, 1}, {1, 0}, {0, 2}});
    check(mixed.entryCount() == 5 && mixed.edgeCount() == 2, "repeated, reciprocal and loop entries counted once");
    checkEqual(neighborLists(mixed), {{1, 2}, {0}, {0}}, "unique loop-free neighbours");

    checkEqual(neighborLists(makeGraph(4, {{0, 3}, {0, 1}, {2, 0}})), {{1, 2, 3}, {0}, {0}, {0}},
               "neighbours in increasing ID order");
    rejects([] { Graph::Builder(3).addEdge(3, 0); }, "entry beyond the last node", "out of range");
}

void loadedFiles() {
    ScratchDirectory scratch;
    const auto symmetric = writeFile(scratch / "symmetric.mtx",
                                     "%%MatrixMarket matrix coordinate pattern symmetric\n4 4 3\n2 1\n3 2\n4 4\n");
    const auto graph = engine::loadGraph(symmetric.string());
    check(graph.nodeCount() == 4 && graph.edgeCount() == 2, "symmetric file: nodes and edges");
    check(graph.entryCount() == 5, "symmetric file: each off-diagonal entry is read both ways, the diagonal once");
    checkEqual(neighborLists(graph), {{1}, {0, 2}, {1}, {}}, "symmetric file neighbours");

    const auto general = writeFile(scratch / "general.mtx",
                                   "%%MatrixMarket matrix coordinate pattern general\n3 3 4\n1 2\n1 2\n2 1\n3 3\n");
    const auto directed = engine::loadGraph(general.string());
    check(directed.nodeCount() == 3 && directed.entryCount() == 4 && directed.edgeCount() == 1, "general file counts");
    checkEqual(neighborLists(directed), {{1}, {0}, {}}, "general file neighbours");

    rejects([&] { engine::loadGraph((scratch / "missing.mtx").string()); }, "missing file", "Cannot open");
}

// One edge written four ways: 0->1, 1->0, both, and both repeated.
void singleEdgeEncodings() {
    const std::vector<Entries> encodings = {
        {{0, 1}}, {{1, 0}}, {{0, 1}, {1, 0}}, {{0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}}};
    const auto first = makeGraph(2, encodings[0]);
    for (const auto& algorithm : kRankedAlgorithms) {
        checkEqual(candidates(first, algorithm, 2), {0, 1}, "single edge: " + algorithm);
    }
    const auto pageRank = engine::pageRankScores(first);
    check(std::abs(pageRank[0] - 0.5) <= 1e-14 && std::abs(pageRank[1] - 0.5) <= 1e-14, "single edge: PageRank 0.5");
    checkEqual(engine::betweennessScores(first, 1), {0, 0}, "single edge: Betweenness");
    for (size_t index = 1; index < encodings.size(); ++index) {
        const auto graph = makeGraph(2, encodings[index]);
        const auto label = "single edge, encoding " + std::to_string(index);
        check(graph.edgeCount() == 1, label + ": one edge");
        checkEqual(neighborLists(graph), neighborLists(first), label + ": neighbours");
        checkEquivalent(graph, first, 2, label);
    }
    for (uint32_t seed = 0; seed < 2; ++seed) {
        std::vector<Owner> ownership(2, Owner::Neutral);
        ownership[seed] = Owner::A;
        const auto result = engine::spread(first, ownership, 1.0);
        check(result.rounds == 1 && ownership == std::vector<Owner>{Owner::A, Owner::A},
              "single edge: the spread crosses it from node " + std::to_string(seed));
    }
}

void selfLoops() {
    const auto plain = makeGraph(2, {{0, 1}});
    Entries entries = {{0, 1}};
    entries.insert(entries.end(), 8, {1, 1});
    const auto looped = makeGraph(2, entries);
    check(looped.entryCount() == 9 && looped.edgeCount() == plain.edgeCount(), "self-loops add no edge");
    checkEqual(neighborLists(looped), neighborLists(plain), "self-loops: neighbours");
    checkEquivalent(looped, plain, 2, "self-loops");
}

void distinctTopologies() {
    const auto middleOne = makeGraph(3, {{0, 1}, {1, 2}});
    const auto middleTwo = makeGraph(3, {{0, 2}, {2, 1}});
    check(middleOne.entryCount() == middleTwo.entryCount() && middleOne.edgeCount() == middleTwo.edgeCount(),
          "both paths have the same counts");
    check(neighborLists(middleOne) != neighborLists(middleTwo), "the two paths have different neighbours");
    checkEqual(engine::betweennessScores(middleOne, 1), {0, 2, 0}, "Betweenness of the path 0-1-2");
    checkEqual(engine::betweennessScores(middleTwo, 1), {0, 0, 2}, "Betweenness of the path 0-2-1");
    checkEqual(engine::pageRankCandidates(middleTwo, 1), {2}, "PageRank of the path 0-2-1");
}

void smallRelation() {
    const Entries entries = {{0, 1}, {0, 1}, {1, 0}, {2, 1}, {2, 2}, {3, 0}};
    const auto raw = makeGraph(4, entries);
    const auto canonical = bothWays(symmetrize(4, entries));
    checkEqual(neighborLists(raw), {{1, 3}, {0, 2}, {1}, {0}}, "neighbours");
    checkEqual(neighborLists(canonical), neighborLists(raw), "neighbours of the both-ways encoding");
    checkEquivalent(raw, canonical, 4, "small relation");

    std::vector<Owner> rawOwnership = {Owner::Neutral, Owner::Neutral, Owner::A, Owner::B};
    auto canonicalOwnership = rawOwnership;
    const auto rawResult = engine::spread(raw, rawOwnership, 0.5);
    const auto canonicalResult = engine::spread(canonical, canonicalOwnership, 0.5);
    check(rawResult.rounds == canonicalResult.rounds && rawResult.stopReason == canonicalResult.stopReason,
          "spread rounds and stop reason");
    checkEqual(rawOwnership, canonicalOwnership, "spread ownership");

    const auto path = makeGraph(3, {{0, 1}, {1, 0}, {1, 2}, {2, 1}});
    for (const auto& algorithm : kRankedAlgorithms) {
        checkEqual(candidates(path, algorithm, 3), {1, 0, 2}, "path both ways: " + algorithm);
    }
}

// wiki-Vote is directed, and many of its links are listed one way only: nodes 51, 62 and 81
// (0-based) each appear in a single entry, (51, 53), (62, 5) and (81, 92). The loader must link
// both ends of every entry.
void wikiVote() {
    const std::string file = "data/wiki-Vote.mtx";
    const auto wiki = engine::loadGraph(file);
    const auto relation = symmetrize(wiki.nodeCount(), rawEntries(file));
    checkEqual(relation[51], {53}, "node 51");
    checkEqual(relation[62], {5}, "node 62");
    checkEqual(relation[81], {92}, "node 81");
    check(neighborLists(wiki) == relation, "wiki-Vote neighbours differ from the file's undirected relation");

    const auto canonical = bothWays(relation);
    check(canonical.entryCount() != wiki.entryCount(), "the both-ways encoding has other entries");
    check(neighborLists(canonical) == relation, "the both-ways encoding changed the neighbours");
    checkEquivalent(wiki, canonical, 40, "wiki-Vote");

    std::vector<Owner> wikiOwnership(wiki.nodeCount(), Owner::Neutral);
    wikiOwnership[51] = Owner::A;
    wikiOwnership[62] = Owner::B;
    auto canonicalOwnership = wikiOwnership;
    const auto wikiResult = engine::spread(wiki, wikiOwnership, 0.5);
    const auto canonicalResult = engine::spread(canonical, canonicalOwnership, 0.5);
    check(wikiResult.rounds == canonicalResult.rounds && wikiOwnership == canonicalOwnership,
          "wiki-Vote spread differs between the encodings");
}

}  // namespace

int main() {
    return runTests({
        {"builder", builder},
        {"loaded files", loadedFiles},
        {"single edge encodings", singleEdgeEncodings},
        {"self-loops", selfLoops},
        {"distinct topologies", distinctTopologies},
        {"small relation", smallRelation},
        {"wiki-Vote", wikiVote},
    });
}
