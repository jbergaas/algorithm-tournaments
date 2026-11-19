#pragma once

// A batch: every match of one network, budget and threshold. Its config is a `key = value` file
// (`#` starts a comment) with exactly these keys, all required:
//   graph = data/ca-GrQc.mtx
//   algorithms = random, degree, coverage_greedy, pagerank, betweenness
//   budget = 20                    # seeds per player; each algorithm lists 2 * budget candidates
//   runs = 100                     # runs of each pair with Random (run IDs 0..runs-1)
//   deterministic_runs = 2         # runs of each pair without Random
//   theta = 0.25                   # threshold in [0, 1]
//   sequences = sequential_a, sequential_b, alternating_a, snake
//   base_seed = 42
//   betweenness_ranking = rankings/ca-GrQc/fullranking.u32le   # only when betweenness is listed
// Every ordered pair of algorithms (self-play included) is played under every sequence.

#include "algorithms.hpp"
#include "betweenness_ranking.hpp"
#include "dataset_identity.hpp"
#include "game.hpp"
#include "graph.hpp"
#include "seeds.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

struct BatchConfig {
    std::string graph;
    std::vector<std::string> algorithms;
    uint32_t budget = 0;
    uint32_t runs = 0;
    uint32_t deterministicRuns = 0;
    double theta = 0;
    std::vector<Sequence> sequences;
    uint64_t baseSeed = 0;
    std::string betweennessRanking;
};

namespace batch_detail {

inline std::string trim(const std::string& text) {
    const auto start = text.find_first_not_of(" \t\r");
    if (start == std::string::npos) return "";
    return text.substr(start, text.find_last_not_of(" \t\r") - start + 1);
}

// Comma-separated items; an empty item (as in "a,,b" or "a,") is kept, so it is rejected later.
inline std::vector<std::string> splitList(const std::string& text) {
    std::vector<std::string> items;
    size_t start = 0;
    for (size_t comma = text.find(','); comma != std::string::npos; comma = text.find(',', start)) {
        items.push_back(trim(text.substr(start, comma - start)));
        start = comma + 1;
    }
    items.push_back(trim(text.substr(start)));
    return items;
}

template <class Integer>
Integer parseInteger(const std::string& key, const std::string& text, Integer minimum) {
    Integer value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < minimum) {
        throw std::invalid_argument(key + " must be an integer of at least " + std::to_string(minimum) + ": " + text);
    }
    return value;
}

inline double parseTheta(const std::string& text) {
    size_t used = 0;
    double value = -1;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
    }
    if (used != text.size() || !(value >= 0 && value <= 1)) {
        throw std::invalid_argument("theta must be a number in [0, 1]: " + text);
    }
    return value;
}

inline std::map<std::string, std::string> readKeyValues(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open config: " + path);
    std::map<std::string, std::string> values;
    int lineNumber = 0;
    for (std::string line; std::getline(file, line);) {
        ++lineNumber;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::invalid_argument(path + ":" + std::to_string(lineNumber) + ": expected key = value");
        }
        const auto key = trim(line.substr(0, equals));
        if (!values.emplace(key, trim(line.substr(equals + 1))).second) {
            throw std::invalid_argument(path + ": key " + key + " given twice");
        }
    }
    return values;
}

}  // namespace batch_detail

inline BatchConfig parseBatchConfig(const std::map<std::string, std::string>& values) {
    using namespace batch_detail;
    std::set<std::string> expected = {"graph", "algorithms", "budget", "runs", "deterministic_runs",
                                      "theta", "sequences", "base_seed"};
    auto requireKey = [&](const std::string& key) {
        if (!values.count(key) || values.at(key).empty()) throw std::invalid_argument("missing key: " + key);
    };
    for (const auto& key : expected) requireKey(key);
    BatchConfig config;
    config.algorithms = splitList(values.at("algorithms"));
    for (const auto& algorithm : config.algorithms) {
        if (std::find(kAlgorithms.begin(), kAlgorithms.end(), algorithm) == kAlgorithms.end()) {
            throw std::invalid_argument("unknown algorithm: " + algorithm);
        }
        if (std::count(config.algorithms.begin(), config.algorithms.end(), algorithm) > 1) {
            throw std::invalid_argument("algorithm listed twice: " + algorithm);
        }
    }
    if (std::count(config.algorithms.begin(), config.algorithms.end(), "betweenness")) {
        expected.insert("betweenness_ranking");
        requireKey("betweenness_ranking");
    }
    for (const auto& [key, value] : values) {
        if (!expected.count(key)) throw std::invalid_argument("unexpected key: " + key);
    }
    config.graph = values.at("graph");
    config.budget = parseInteger<uint32_t>("budget", values.at("budget"), 1);
    config.runs = parseInteger<uint32_t>("runs", values.at("runs"), 1);
    config.deterministicRuns = parseInteger<uint32_t>("deterministic_runs", values.at("deterministic_runs"), 1);
    if (config.deterministicRuns > config.runs) throw std::invalid_argument("deterministic_runs exceeds runs");
    config.theta = parseTheta(values.at("theta"));
    for (const auto& name : splitList(values.at("sequences"))) {
        const auto sequence = parseSequence(name);
        if (std::count(config.sequences.begin(), config.sequences.end(), sequence)) {
            throw std::invalid_argument("sequence listed twice: " + name);
        }
        config.sequences.push_back(sequence);
    }
    config.baseSeed = parseInteger<uint64_t>("base_seed", values.at("base_seed"), 0);
    if (values.count("betweenness_ranking")) config.betweennessRanking = values.at("betweenness_ranking");
    return config;
}

inline BatchConfig loadBatchConfig(const std::string& path) {
    return parseBatchConfig(batch_detail::readKeyValues(path));
}

struct PlannedMatch {
    std::string algorithmA;
    std::string algorithmB;
    Sequence sequence;
    uint32_t run;
};

// The matches of a batch in the order they are played and written: by A's algorithm, B's
// algorithm, sequence and run, each in config order.
inline std::vector<PlannedMatch> plannedMatches(const BatchConfig& config) {
    std::vector<PlannedMatch> matches;
    for (const auto& a : config.algorithms) {
        for (const auto& b : config.algorithms) {
            const uint32_t runs = a == "random" || b == "random" ? config.runs : config.deterministicRuns;
            for (const auto sequence : config.sequences) {
                for (uint32_t run = 0; run < runs; ++run) matches.push_back({a, b, sequence, run});
            }
        }
    }
    return matches;
}

namespace batch_detail {

inline std::string csvField(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string quoted = "\"";
    for (char c : value) {
        if (c == '"') quoted += '"';
        quoted += c;
    }
    return quoted + '"';
}

inline std::string jsonList(const std::vector<uint32_t>& nodes) {
    std::string text = "[";
    for (size_t i = 0; i < nodes.size(); ++i) text += (i ? "," : "") + std::to_string(nodes[i]);
    return text + "]";
}

// Candidates of every algorithm except Random, which draws new ones for every run.
inline std::map<std::string, std::vector<uint32_t>> fixedCandidates(const BatchConfig& config, const Graph& graph) {
    const uint32_t count = 2 * config.budget;
    std::map<std::string, std::vector<uint32_t>> candidates;
    for (const auto& algorithm : config.algorithms) {
        if (algorithm == "degree") candidates[algorithm] = degreeCandidates(graph, count);
        if (algorithm == "coverage_greedy") candidates[algorithm] = coverageGreedyCandidates(graph, count);
        if (algorithm == "pagerank") candidates[algorithm] = pageRankCandidates(graph, count);
        if (algorithm == "betweenness") {
            auto ranking = loadBetweennessRanking(config.betweennessRanking, graph.nodeCount());
            ranking.resize(count);
            candidates[algorithm] = ranking;
        }
    }
    return candidates;
}

}  // namespace batch_detail

// Plays every match of the batch and writes one row per match, in plannedMatches() order, to
// `resultsPath` (outcome) and `tracePath` (both candidate lists and both seed sets).
inline size_t runBatch(const BatchConfig& config, const std::string& resultsPath, const std::string& tracePath) {
    using namespace batch_detail;
    const Graph graph = loadGraph(config.graph);
    if (2 * static_cast<uint64_t>(config.budget) > graph.nodeCount()) {
        throw std::invalid_argument("twice the budget exceeds the number of nodes");
    }
    const std::string identity = computeDatasetIdentity(config.graph, graph.nodeCount());
    const auto fixed = fixedCandidates(config, graph);
    const uint32_t k = config.budget;
    const std::string name = csvField(std::filesystem::path(config.graph).filename().string());
    std::ostringstream thetaText;
    thetaText << std::setprecision(std::numeric_limits<double>::max_digits10) << config.theta;

    std::ofstream results(resultsPath);
    std::ofstream trace(tracePath);
    results << "graph,nodes,edges,algorithm_a,algorithm_b,seed_count,run_id,rng_seed_a,rng_seed_b,draft_order,"
               "theta,spread_a,spread_b,neutral,ranking_overlap,skipped_a,skipped_b,rounds,stop_reason\n";
    trace << "graph,algorithm_a,algorithm_b,seed_count,run_id,rng_seed_a,rng_seed_b,draft_order,theta,"
             "ranking_a,ranking_b,seeds_a,seeds_b\n";
    const auto matches = plannedMatches(config);
    for (const auto& match : matches) {
        const uint64_t seedA = matchSeed(config.baseSeed, identity, match.algorithmA, k, match.run, 'A');
        const uint64_t seedB = matchSeed(config.baseSeed, identity, match.algorithmB, k, match.run, 'B');
        const auto rankingA = match.algorithmA == "random" ? randomCandidates(graph, 2 * k, seedA) : fixed.at(match.algorithmA);
        const auto rankingB = match.algorithmB == "random" ? randomCandidates(graph, 2 * k, seedB) : fixed.at(match.algorithmB);
        const auto outcome = playMatch(graph, rankingA, rankingB, k, match.sequence, config.theta);
        const auto& allocation = outcome.allocation;
        const std::string key = csvField(match.algorithmA) + ',' + csvField(match.algorithmB) + ','
            + std::to_string(k) + ',' + std::to_string(match.run) + ',' + std::to_string(seedA) + ','
            + std::to_string(seedB) + ',' + sequenceName(match.sequence) + ',' + thetaText.str();
        results << name << ',' << graph.nodeCount() << ',' << graph.entryCount() << ',' << key << ','
                << outcome.spreadA << ',' << outcome.spreadB << ',' << outcome.neutral << ','
                << allocation.rankingOverlap << ',' << allocation.skippedA << ',' << allocation.skippedB << ','
                << outcome.spread.rounds << ',' << stopReasonName(outcome.spread.stopReason) << '\n';
        trace << name << ',' << key << ',' << csvField(jsonList(rankingA)) << ',' << csvField(jsonList(rankingB))
              << ',' << csvField(jsonList(allocation.seedsA)) << ',' << csvField(jsonList(allocation.seedsB)) << '\n';
    }
    if (!results.flush() || !trace.flush()) throw std::runtime_error("Cannot write the batch output");
    return matches.size();
}

}  // namespace engine
