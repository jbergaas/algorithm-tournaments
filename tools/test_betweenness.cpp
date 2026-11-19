// The parallel Betweenness (algorithms.hpp) against the single-thread reference
// (reference_betweenness.hpp): bitwise equal scores and the same ranking for 1, 2, 4 and 8
// threads. Also hand-computed scores, the tie order of rankByScore, and the stored ranking files
// (betweenness_ranking.hpp). With `--real-graphs FILE...` it compares the two on real networks
// instead (at most 15,000 nodes, so the reference stays fast).

#include "check.hpp"
#include "reference_betweenness.hpp"
#include "engine/algorithms.hpp"
#include "engine/betweenness_ranking.hpp"
#include "engine/graph.hpp"

#include <omp.h>

#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Entries = std::vector<std::pair<uint32_t, uint32_t>>;

constexpr std::array<int, 4> kThreadCounts = {1, 2, 4, 8};

engine::Graph makeGraph(uint32_t nodeCount, const Entries& entries) {
    engine::Graph::Builder builder(nodeCount);
    for (const auto& [from, to] : entries) builder.addEdge(from, to);
    return builder.build();
}

engine::Graph path(uint32_t nodeCount) {
    Entries entries;
    for (uint32_t node = 1; node < nodeCount; ++node) entries.emplace_back(node - 1, node);
    return makeGraph(nodeCount, entries);
}

void checkSameBits(const std::vector<double>& actual, const std::vector<double>& expected, const std::string& label) {
    check(actual.size() == expected.size(), label + ": score count differs");
    size_t node = 0;
    while (node < actual.size() && std::bit_cast<uint64_t>(actual[node]) == std::bit_cast<uint64_t>(expected[node])) {
        ++node;
    }
    check(node == actual.size(), label + ": score bits differ at node " + std::to_string(node));
}

void compareWithReference(const engine::Graph& graph, const std::string& label) {
    const auto expected = reference::betweenness(graph);
    const auto ranking = reference::ranking(expected);
    for (int threads : kThreadCounts) {
        const auto context = label + ", " + std::to_string(threads) + " threads";
        const auto scores = engine::betweennessScores(graph, threads);
        checkSameBits(scores, expected, context);
        checkEqual(engine::rankByScore(scores), ranking, context + ": ranking");
    }
}

void smallGraphs() {
    compareWithReference(makeGraph(0, {}), "no nodes");
    compareWithReference(makeGraph(1, {}), "one node");
    compareWithReference(path(9), "path of 9");
    compareWithReference(makeGraph(6, {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}}), "star");
    compareWithReference(makeGraph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}}), "cycle");
    compareWithReference(makeGraph(5, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {1, 2}}), "diamond with chord");
    compareWithReference(makeGraph(10, {{0, 1}, {1, 2}, {3, 4}, {4, 5}, {5, 3}, {6, 7}}), "components");
    compareWithReference(makeGraph(8, {{0, 1}, {0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 1}, {3, 3}, {4, 5}}),
                         "repeated entries and self-loops");
    const Entries twoCycles = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 5}};
    compareWithReference(makeGraph(10, twoCycles), "two 5-cycles");
    compareWithReference(path(64), "path of 64");
}

void randomGraphs() {
    std::mt19937 random(7182);
    for (uint32_t n = 5; n < 23; ++n) {
        Entries entries;
        for (uint32_t from = 0; from < n; ++from) {
            for (uint32_t to = 0; to < n; ++to) {
                if (random() % 11 == 0) entries.emplace_back(from, to);
            }
        }
        compareWithReference(makeGraph(n, entries), "random graph of " + std::to_string(n) + " nodes");
    }
}

void handComputedScores() {
    for (int threads : kThreadCounts) {
        const auto label = std::to_string(threads) + " threads: ";
        checkEqual(engine::betweennessScores(path(3), threads), {0, 2, 0}, label + "path of 3");
        checkEqual(engine::betweennessScores(makeGraph(5, {{0, 1}, {0, 2}, {0, 3}, {0, 4}}), threads),
                   {12, 0, 0, 0, 0}, label + "star");
        checkEqual(engine::betweennessScores(makeGraph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}}), threads),
                   {1, 1, 1, 1}, label + "cycle of 4");
        checkEqual(engine::betweennessScores(makeGraph(4, {{0, 1}, {1, 2}}), threads), {0, 2, 0, 0},
                   label + "path of 3 and an isolated node");
    }
    checkEqual(engine::rankByScore({1, 2, 2, 0}), {1, 2, 0, 3}, "equal scores by increasing ID");
    checkEqual(engine::rankByScore({0, 0, 0}), {0, 1, 2}, "all scores equal");
}

// The comparisons above only mean something if OpenMP gives each run the threads it asks for.
void requestedTeamSizeIsUsed() {
    for (int threads : kThreadCounts) {
        int observed = 0;
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            observed = omp_get_num_threads();
        }
        check(observed == threads, "OpenMP gave " + std::to_string(observed) + " threads instead of "
                                       + std::to_string(threads));
    }
}

std::vector<double> readScores(const std::filesystem::path& file) {
    const auto bytes = readFile(file);
    check(bytes.size() % 8 == 0, "score file size is not a multiple of 8");
    std::vector<double> scores(bytes.size() / 8);
    for (size_t index = 0; index < scores.size(); ++index) {
        uint64_t bits = 0;
        for (unsigned byte = 0; byte < 8; ++byte) {
            bits |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[8 * index + byte])) << (8 * byte);
        }
        scores[index] = std::bit_cast<double>(bits);
    }
    return scores;
}

void storedRanking() {
    ScratchDirectory scratch;
    Entries entries;
    for (uint32_t node = 1; node < 30; ++node) entries.emplace_back(node - 1, node);
    entries.insert(entries.end(), {{3, 9}, {3, 10}, {9, 11}, {10, 11}, {32, 33}, {33, 34}});
    const auto graph = makeGraph(40, entries);
    const auto expected = reference::betweenness(graph);
    const auto directory = scratch / "ranking";
    engine::writeBetweennessRanking(graph, 4, directory);
    checkSameBits(readScores(directory / "scores.f64le"), expected, "stored scores");
    const auto file = (directory / "fullranking.u32le").string();
    checkEqual(engine::loadBetweennessRanking(file, 40), reference::ranking(expected), "stored ranking");

    rejects([&] { engine::loadBetweennessRanking(file, 41); }, "ranking of 40 nodes for 41", "one entry per node");
    rejects([&] { engine::loadBetweennessRanking(file, 39); }, "ranking of 40 nodes for 39", "one entry per node");
    const auto write = [&](const std::string& name, const std::vector<uint32_t>& nodes) {
        return writeFile(scratch / name, littleEndian(nodes)).string();
    };
    rejects([&] { engine::loadBetweennessRanking(write("duplicate", {0, 1, 1}), 3); }, "repeated node",
            "not an ordering");
    rejects([&] { engine::loadBetweennessRanking(write("range", {0, 1, 3}), 3); }, "node out of range",
            "not an ordering");
    rejects([&] { engine::loadBetweennessRanking((scratch / "missing").string(), 3); }, "missing file",
            "Cannot open");
    checkEqual(engine::loadBetweennessRanking(write("valid", {2, 0, 1}), 3), {2, 0, 1}, "valid ranking");
}

void realGraphs(const std::vector<std::string>& files) {
    check(!files.empty(), "--real-graphs needs at least one file");
    for (const auto& file : files) {
        const auto graph = engine::loadGraph(file);
        check(graph.nodeCount() <= 15000, file + " has more than 15,000 nodes");
        compareWithReference(graph, file);
        std::cout << file << ": " << graph.nodeCount()
                  << " nodes, scores and ranking equal to the reference for 1, 2, 4 and 8 threads\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--real-graphs") {
        const std::vector<std::string> files(argv + 2, argv + argc);
        return runTests({{"real graphs", [&] { realGraphs(files); }}});
    }
    return runTests({
        {"small graphs", smallGraphs},
        {"random graphs", randomGraphs},
        {"hand-computed scores", handComputedScores},
        {"requested team size", requestedTeamSizeIsUsed},
        {"stored ranking", storedRanking},
    });
}
