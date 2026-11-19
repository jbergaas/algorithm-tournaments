// Known answers of the seed-selection algorithms (algorithms.hpp): Random, Degree, Coverage
// Greedy and PageRank, and the candidate lists all five produce. Betweenness's scores are
// checked in test_betweenness.cpp.

#include "check.hpp"
#include "engine/algorithms.hpp"
#include "engine/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::Graph;
using Nodes = std::vector<uint32_t>;
using Entries = std::vector<std::pair<uint32_t, uint32_t>>;

Graph makeGraph(uint32_t nodeCount, const Entries& entries) {
    Graph::Builder builder(nodeCount);
    for (const auto& [from, to] : entries) builder.addEdge(from, to);
    return builder.build();
}

// Neighbours: 0 {1,2,3,4}, 1 {0,2,3}, 2 {0,1,4,5}, 3 {0,1,5}, 4 {0,2,5}, 5 {2,3,4}.
Graph overlapGraph() {
    return makeGraph(6, {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 5}, {4, 5}});
}

Nodes candidates(const Graph& graph, const std::string& algorithm, uint32_t count) {
    if (algorithm == "random") return engine::randomCandidates(graph, count, 12345);
    if (algorithm == "degree") return engine::degreeCandidates(graph, count);
    if (algorithm == "coverage_greedy") return engine::coverageGreedyCandidates(graph, count);
    if (algorithm == "pagerank") return engine::pageRankCandidates(graph, count);
    auto ranking = engine::rankByScore(engine::betweennessScores(graph, 1));
    ranking.resize(std::min<size_t>(count, ranking.size()));
    return ranking;
}

Nodes prefix(const Nodes& nodes, uint32_t count) {
    return {nodes.begin(), nodes.begin() + std::min<size_t>(count, nodes.size())};
}

// Every algorithm lists each node at most once, and a shorter list is the start of a longer one.
void candidateLists() {
    const auto graph = overlapGraph();
    for (const auto& algorithm : engine::kAlgorithms) {
        const auto all = candidates(graph, algorithm, 99);
        check(all.size() == 6 && std::set<uint32_t>(all.begin(), all.end()).size() == 6,
              algorithm + ": a list of all nodes when more are asked for");
        check(*std::max_element(all.begin(), all.end()) < 6, algorithm + ": node out of range");
        for (uint32_t count : {0U, 1U, 3U, 6U}) {
            checkEqual(candidates(graph, algorithm, count), prefix(all, count),
                       algorithm + ": first " + std::to_string(count));
        }
    }
}

void degree() {
    checkEqual(engine::degreeCandidates(overlapGraph(), 6), {0, 2, 1, 3, 4, 5}, "higher degree first, ties by ID");
}

void coverageGreedy() {
    // After node 0, node 2 adds two uncovered nodes (5 and 0), as do 3 and 4, and has the higher
    // degree; node 1 adds only node 0. Then nothing is left to cover and degree, then ID, decide.
    checkEqual(engine::coverageGreedyCandidates(overlapGraph(), 6), {0, 2, 1, 3, 4, 5}, "overlap graph");

    // Node 0 covers 1..5. Then 5, 6, 7, 8 and 9 all add two nodes, and 9 has the highest degree.
    // After 5 and 7, only degree and ID order the rest: 6 and 8 (degree 2), then 1..4.
    const auto equalGains = makeGraph(10, {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {6, 7}, {6, 8}, {9, 7}, {9, 8},
                                           {9, 5}});
    checkEqual(engine::coverageGreedyCandidates(equalGains, 10), {0, 9, 5, 7, 6, 8, 1, 2, 3, 4},
               "equal gains by higher degree");

    // In a triangle every node covers the others, so after node 0 the other two still have a
    // gain of 1 (node 0 itself); the 45 isolated nodes follow by ID.
    const auto triangle = makeGraph(48, {{0, 1}, {1, 2}, {2, 0}});
    Nodes byId(48);
    std::iota(byId.begin(), byId.end(), 0);
    checkEqual(engine::coverageGreedyCandidates(triangle, 48), byId, "triangle and isolated nodes");
    for (uint32_t count : {40U, 10U, 20U, 30U}) {
        checkEqual(engine::coverageGreedyCandidates(triangle, count), prefix(byId, count),
                   "triangle and isolated nodes, first " + std::to_string(count));
    }
}

void pageRank() {
    // Entries 0->1, 1->2, 2->1, 3->1: undirected, node 1 is the centre of a star.
    const auto hub = makeGraph(4, {{0, 1}, {1, 2}, {2, 1}, {3, 1}});
    checkEqual(engine::pageRankCandidates(hub, 4), {1, 0, 2, 3}, "star centre first, leaves by ID");

    const auto scores = engine::pageRankScores(makeGraph(3, {}));
    check(std::all_of(scores.begin(), scores.end(), [](double s) { return std::abs(s - 1.0 / 3) <= 1e-15; }),
          "nodes without neighbours share the rank evenly");
    checkEqual(engine::pageRankCandidates(makeGraph(3, {}), 3), {0, 1, 2}, "equal scores by ID");

    const auto pathAndIsolated = engine::pageRankScores(makeGraph(4, {{0, 1}, {1, 2}}));
    check(pathAndIsolated[0] == pathAndIsolated[2], "the two ends of a path score the same");
    check(pathAndIsolated[1] > pathAndIsolated[0] && pathAndIsolated[0] > pathAndIsolated[3],
          "middle above ends above the isolated node");
    const double total = std::accumulate(pathAndIsolated.begin(), pathAndIsolated.end(), 0.0);
    check(std::abs(total - 1.0) <= 1e-12, "the scores sum to 1");
}

// The study ranks by 100 iterations of PageRank; 500 give the same top 40 on the six networks.
void pageRankIterations() {
    for (const std::string file : {"data/Erdos02.mtx", "data/ca-GrQc.mtx", "data/ca-HepPh.mtx", "data/ca-CondMat.mtx",
                                   "data/wiki-Vote.mtx", "data/email-Enron.mtx"}) {
        const auto graph = engine::loadGraph(file);
        const auto ranking500 = engine::rankByScore(engine::pageRankScores(graph, 500));
        checkEqual(engine::pageRankCandidates(graph, 40), prefix(ranking500, 40),
                   file + ": top 40 after 100 and 500 iterations");
    }
}

void randomDraws() {
    Graph::Builder builder(32);
    for (uint32_t node = 1; node < 32; ++node) builder.addEdge(node - 1, node);
    const auto path = builder.build();
    check(engine::randomCandidates(path, 12, 111) == engine::randomCandidates(path, 12, 111), "same seed, same draw");
    check(engine::randomCandidates(path, 12, 11) != engine::randomCandidates(path, 12, 29), "other seed, other draw");
    check(engine::randomCandidates(path, 12, 1) != engine::randomCandidates(path, 12, (1ULL << 40) | 1),
          "the high 32 bits of the seed count");

    // Player A's draw in run 0 of the pinned example in test_seeds.cpp, on the ten-node test graph.
    // std::shuffle differs between standard libraries. These are the values of libstdc++ (GCC),
    // which the study used.
    const auto testGraph = engine::loadGraph("data/test.mtx");
    checkEqual(engine::randomCandidates(testGraph, 10, 14454494734286381522ULL), {5, 0, 4, 9, 8, 1, 2, 6, 3, 7},
               "pinned draw");
}

}  // namespace

int main() {
    return runTests({
        {"candidate lists", candidateLists},
        {"degree", degree},
        {"coverage greedy", coverageGreedy},
        {"pagerank", pageRank},
        {"pagerank iterations", pageRankIterations},
        {"random", randomDraws},
    });
}
