// Batches (batch.hpp): the config keys and values, the planned matches, and the two files runBatch
// writes, with one row per planned match: Random's seeds, the candidates played and the outcome.

#include "check.hpp"
#include "engine/algorithms.hpp"
#include "engine/batch.hpp"
#include "engine/betweenness_ranking.hpp"
#include "engine/dataset_identity.hpp"
#include "engine/game.hpp"
#include "engine/graph.hpp"
#include "engine/seeds.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Nodes = std::vector<uint32_t>;
using Row = std::vector<std::string>;
using Values = std::map<std::string, std::string>;

const std::string kResultsHeader =
    "graph,nodes,edges,algorithm_a,algorithm_b,seed_count,run_id,rng_seed_a,rng_seed_b,draft_order,theta,"
    "spread_a,spread_b,neutral,ranking_overlap,skipped_a,skipped_b,rounds,stop_reason\n";
const std::string kTraceHeader =
    "graph,algorithm_a,algorithm_b,seed_count,run_id,rng_seed_a,rng_seed_b,draft_order,theta,"
    "ranking_a,ranking_b,seeds_a,seeds_b\n";

Values validValues() {
    return {{"graph", "graph.mtx"},
            {"algorithms", "random, degree, betweenness"},
            {"budget", "3"},
            {"runs", "3"},
            {"deterministic_runs", "1"},
            {"theta", "0.375"},
            {"sequences", "sequential_a, snake"},
            {"base_seed", "424242"},
            {"betweenness_ranking", "ranking.u32le"}};
}

Values with(Values values, const std::string& key, const std::string& value) {
    values[key] = value;
    return values;
}

Values without(Values values, const std::string& key) {
    values.erase(key);
    return values;
}

bool sameConfig(const engine::BatchConfig& a, const engine::BatchConfig& b) {
    return a.graph == b.graph && a.algorithms == b.algorithms && a.budget == b.budget && a.runs == b.runs
        && a.deterministicRuns == b.deterministicRuns && a.theta == b.theta && a.sequences == b.sequences
        && a.baseSeed == b.baseSeed && a.betweennessRanking == b.betweennessRanking;
}

void configKeys() {
    const auto config = engine::parseBatchConfig(validValues());
    check(config.graph == "graph.mtx" && config.budget == 3 && config.runs == 3 && config.deterministicRuns == 1
              && config.theta == 0.375 && config.baseSeed == 424242 && config.betweennessRanking == "ranking.u32le",
          "values of a valid config");
    check(config.algorithms == std::vector<std::string>{"random", "degree", "betweenness"}, "algorithms in order");
    check(config.sequences == std::vector{engine::Sequence::SequentialA, engine::Sequence::Snake},
          "sequences in order");

    for (const auto& [key, value] : validValues()) {
        rejects([&] { engine::parseBatchConfig(without(validValues(), key)); }, "without " + key,
                "missing key: " + key);
        rejects([&] { engine::parseBatchConfig(with(validValues(), key, "")); }, "empty " + key, "missing key: " + key);
    }
    rejects([] { engine::parseBatchConfig(with(validValues(), "thetha", "0.5")); }, "unknown key",
            "unexpected key: thetha");

    const auto withoutBetweenness = with(validValues(), "algorithms", "random, degree");
    rejects([&] { engine::parseBatchConfig(withoutBetweenness); }, "betweenness_ranking without betweenness",
            "unexpected key: betweenness_ranking");
    const auto plain = engine::parseBatchConfig(without(withoutBetweenness, "betweenness_ranking"));
    check(plain.betweennessRanking.empty(), "no betweenness_ranking needed without betweenness");
}

void configValues() {
    const auto rejected = [](const std::string& key, const std::string& value, const std::string& expected) {
        rejects([&] { engine::parseBatchConfig(with(validValues(), key, value)); }, key + " = " + value, expected);
    };
    for (const std::string value : {"0", "-1", "+1", "1x", "1.5", "4294967296"}) {
        rejected("budget", value, "budget must be");
        rejected("runs", value, "runs must be");
    }
    rejected("deterministic_runs", "0", "deterministic_runs must be");
    rejected("deterministic_runs", "4", "deterministic_runs exceeds runs");
    for (const std::string value : {"-0.01", "1.01", "nan", "inf", "-inf", "abc", "0.5 trailing"}) {
        rejected("theta", value, "theta must be");
    }
    for (const std::string value : {"-1", "+1", "42x", "18446744073709551616"}) {
        rejected("base_seed", value, "base_seed must be");
    }
    rejected("sequences", "unknown", "unknown sequence");
    rejected("sequences", "snake, snake", "sequence listed twice");
    rejected("sequences", "snake,", "unknown sequence");
    rejected("algorithms", "random, unknown, betweenness", "unknown algorithm");
    rejected("algorithms", "degree, betweenness, degree", "algorithm listed twice");
    rejected("algorithms", "random,, degree", "unknown algorithm");

    const auto parse = [](const std::string& key, const std::string& value) {
        return engine::parseBatchConfig(with(validValues(), key, value));
    };
    check(parse("theta", "0").theta == 0.0 && parse("theta", "0.5").theta == 0.5 && parse("theta", "1").theta == 1.0,
          "theta 0, 0.5 and 1");
    check(parse("budget", "4294967295").budget == 4294967295U, "largest budget");
    check(parse("base_seed", "0").baseSeed == 0 && parse("base_seed", "18446744073709551615").baseSeed == UINT64_MAX,
          "smallest and largest base seed");
    check(parse("deterministic_runs", "3").deterministicRuns == 3, "deterministic_runs equal to runs");
}

void configFile() {
    ScratchDirectory scratch;
    const std::string text =
        "# A batch.\n"
        "graph = graph.mtx\n"
        "\n"
        "algorithms = random, degree, betweenness   # three of them\n"
        "  budget=3\n"
        "runs = 3\n"
        "deterministic_runs = 1\n"
        "theta = 0.375\n"
        "sequences = sequential_a, snake\n"
        "base_seed = 424242\n"
        "betweenness_ranking = ranking.u32le\n";
    const auto file = writeFile(scratch / "batch.cfg", text).string();
    check(sameConfig(engine::loadBatchConfig(file), engine::parseBatchConfig(validValues())),
          "the file gives the same config as its keys");
    const auto twice = writeFile(scratch / "twice.cfg", text + "budget = 3\n").string();
    rejects([&] { engine::loadBatchConfig(twice); }, "key given twice", "key budget given twice");
    const auto noEquals = writeFile(scratch / "no-equals.cfg", text + "budget 3\n").string();
    rejects([&] { engine::loadBatchConfig(noEquals); }, "line without =", "expected key = value");
    rejects([&] { engine::loadBatchConfig((scratch / "missing.cfg").string()); }, "missing file",
            "Cannot open config");
}

// The plan lists the matches by A's algorithm, B's algorithm, sequence and run, each in config
// order, with `runs` runs for pairs with Random and `deterministic_runs` for the others.
void checkPlan(const engine::BatchConfig& config, const std::vector<engine::PlannedMatch>& plan) {
    const auto position = [](const auto& list, const auto& item) {
        return std::find(list.begin(), list.end(), item) - list.begin();
    };
    std::map<std::tuple<std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t>, uint32_t> runsPerGroup;
    std::tuple<std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t> previous = {-1, -1, -1, -1};
    for (const auto& match : plan) {
        const auto a = position(config.algorithms, match.algorithmA);
        const auto b = position(config.algorithms, match.algorithmB);
        const auto sequence = position(config.sequences, match.sequence);
        const auto key = std::tuple{a, b, sequence, static_cast<std::ptrdiff_t>(match.run)};
        check(key > previous, "matches out of order at " + match.algorithmA + " vs " + match.algorithmB);
        previous = key;
        ++runsPerGroup[{a, b, sequence}];
    }
    const size_t algorithms = config.algorithms.size();
    check(runsPerGroup.size() == algorithms * algorithms * config.sequences.size(), "every pair under every sequence");
    for (const auto& [group, runs] : runsPerGroup) {
        const bool withRandom = config.algorithms[std::get<0>(group)] == "random"
                             || config.algorithms[std::get<1>(group)] == "random";
        check(runs == (withRandom ? config.runs : config.deterministicRuns), "runs of a pair");
    }
}

void plannedMatches() {
    const auto full = engine::parseBatchConfig(with(validValues(), "deterministic_runs", "3"));
    const auto reduced = engine::parseBatchConfig(validValues());
    const auto fullPlan = engine::plannedMatches(full);
    const auto reducedPlan = engine::plannedMatches(reduced);
    check(fullPlan.size() == 54, "3 x 3 pairs x 2 sequences x 3 runs");
    check(reducedPlan.size() == 38, "(5 pairs with Random x 3 runs + 4 pairs x 1 run) x 2 sequences");
    checkPlan(full, fullPlan);
    checkPlan(reduced, reducedPlan);
    const auto describe = [](const engine::PlannedMatch& match) {
        return match.algorithmA + " " + match.algorithmB + " " + engine::sequenceName(match.sequence) + " "
             + std::to_string(match.run);
    };
    check(describe(fullPlan[0]) == "random random sequential_a 0", "first match");
    check(describe(fullPlan[3]) == "random random snake 0", "fourth match");
    check(describe(fullPlan[6]) == "random degree sequential_a 0", "seventh match");
    check(describe(fullPlan.back()) == "betweenness betweenness snake 2", "last match");
    check(describe(reducedPlan[18]) == "degree random sequential_a 0", "a pair with Random keeps its runs");
    check(describe(reducedPlan[24]) == "degree degree sequential_a 0"
              && describe(reducedPlan[25]) == "degree degree snake 0",
          "one run of a pair without Random");
}

// The graph's file name goes through csvField; commas, quotes and line breaks are quoted.
void csvFields() {
    check(engine::batch_detail::csvField("ca-GrQc.mtx") == "ca-GrQc.mtx", "plain field");
    check(engine::batch_detail::csvField("a,\"b\"\nc") == "\"a,\"\"b\"\"\nc\"", "field with comma, quotes, newline");
}

// The fields of a CSV line; a quoted field may hold commas and doubled quotes.
Row splitCsv(const std::string& line) {
    Row fields(1);
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted && c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
            fields.back() += '"';
            ++i;
        } else if (c == '"') {
            quoted = !quoted;
        } else if (c == ',' && !quoted) {
            fields.emplace_back();
        } else {
            fields.back() += c;
        }
    }
    return fields;
}

// The rows of a CSV file without its header.
std::vector<Row> readRows(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::vector<Row> rows;
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) rows.push_back(splitCsv(line));
    return rows;
}

Nodes parseList(const std::string& text) {
    check(text.size() >= 2 && text.front() == '[' && text.back() == ']', "not a list: " + text);
    Nodes nodes;
    std::istringstream items(text.substr(1, text.size() - 2));
    for (std::string item; std::getline(items, item, ',');) nodes.push_back(static_cast<uint32_t>(std::stoul(item)));
    return nodes;
}

Nodes readRanking(const std::filesystem::path& path) {
    const auto bytes = readFile(path);
    Nodes nodes(bytes.size() / 4);
    for (size_t index = 0; index < nodes.size(); ++index) {
        for (unsigned byte = 0; byte < 4; ++byte) {
            nodes[index] |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[4 * index + byte])) << (8 * byte);
        }
    }
    return nodes;
}

// A twelve-node graph (a path with a chord and a self-loop) with its stored Betweenness ranking.
struct Fixture {
    ScratchDirectory scratch;
    std::string graph;
    std::string ranking;

    Fixture() {
        const std::string text = "%%MatrixMarket matrix coordinate pattern general\n12 12 13\n"
                                 "1 2\n2 3\n3 4\n4 5\n5 6\n6 7\n7 8\n8 9\n9 10\n10 11\n11 12\n3 9\n5 5\n";
        graph = writeFile(scratch / "graph.mtx", text).string();
        engine::writeBetweennessRanking(engine::loadGraph(graph), 2, scratch / "ranking");
        ranking = (scratch / "ranking" / "fullranking.u32le").string();
    }

    // All five algorithms, both players, three runs each, two sequences: 150 matches.
    engine::BatchConfig config() const {
        return engine::parseBatchConfig({{"graph", graph},
                                         {"algorithms", "random, degree, coverage_greedy, pagerank, betweenness"},
                                         {"budget", "3"},
                                         {"runs", "3"},
                                         {"deterministic_runs", "3"},
                                         {"theta", "0.375"},
                                         {"sequences", "sequential_a, snake"},
                                         {"base_seed", "424242"},
                                         {"betweenness_ranking", ranking}});
    }

    // Runs the batch into NAME-results.csv and NAME-trace.csv and returns their paths.
    std::pair<std::filesystem::path, std::filesystem::path> run(const engine::BatchConfig& config,
                                                                const std::string& name) const {
        const auto results = scratch / (name + "-results.csv");
        const auto trace = scratch / (name + "-trace.csv");
        const size_t count = engine::runBatch(config, results.string(), trace.string());
        check(count == engine::plannedMatches(config).size(), name + ": runBatch counts the planned matches");
        return {results, trace};
    }
};

void batchFiles() {
    const Fixture fixture;
    const auto config = fixture.config();
    const auto plan = engine::plannedMatches(config);
    const auto [resultsPath, tracePath] = fixture.run(config, "batch");
    check(readFile(resultsPath).starts_with(kResultsHeader), "results header");
    check(readFile(tracePath).starts_with(kTraceHeader), "trace header");
    const auto results = readRows(resultsPath);
    const auto trace = readRows(tracePath);
    check(results.size() == plan.size() && trace.size() == plan.size(), "one row per planned match");

    const auto graph = engine::loadGraph(fixture.graph);
    const auto identity = engine::computeDatasetIdentity(fixture.graph, graph.nodeCount());
    const auto stored = readRanking(fixture.ranking);
    const uint32_t k = 3;
    const auto candidates = [&](const std::string& algorithm, uint64_t seed) {
        if (algorithm == "random") return engine::randomCandidates(graph, 2 * k, seed);
        if (algorithm == "degree") return engine::degreeCandidates(graph, 2 * k);
        if (algorithm == "coverage_greedy") return engine::coverageGreedyCandidates(graph, 2 * k);
        if (algorithm == "pagerank") return engine::pageRankCandidates(graph, 2 * k);
        return Nodes(stored.begin(), stored.begin() + 2 * k);
    };
    for (size_t index = 0; index < plan.size(); ++index) {
        const auto& match = plan[index];
        const auto& row = results[index];
        const auto& traced = trace[index];
        const auto label = "row " + std::to_string(index + 1);
        check(row.size() == 19 && traced.size() == 13, label + ": number of fields");
        checkEqual(Row(row.begin(), row.begin() + 3), {"graph.mtx", "12", std::to_string(graph.entryCount())},
                   label + ": graph, nodes and entries");
        const uint64_t seedA = engine::matchSeed(424242, identity, match.algorithmA, k, match.run, 'A');
        const uint64_t seedB = engine::matchSeed(424242, identity, match.algorithmB, k, match.run, 'B');
        checkEqual(Row(row.begin() + 3, row.begin() + 11),
                   {match.algorithmA, match.algorithmB, "3", std::to_string(match.run), std::to_string(seedA),
                    std::to_string(seedB), engine::sequenceName(match.sequence), "0.375"},
                   label + ": the planned match and its seeds");
        checkEqual(Row(traced.begin(), traced.begin() + 9),
                   {row[0], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]},
                   label + ": the trace row is another match");

        const auto rankingA = parseList(traced[9]);
        const auto rankingB = parseList(traced[10]);
        checkEqual(rankingA, candidates(match.algorithmA, seedA), label + ": candidates of A");
        checkEqual(rankingB, candidates(match.algorithmB, seedB), label + ": candidates of B");
        const auto outcome = engine::playMatch(graph, rankingA, rankingB, k, match.sequence, 0.375);
        checkEqual(parseList(traced[11]), outcome.allocation.seedsA, label + ": seeds of A");
        checkEqual(parseList(traced[12]), outcome.allocation.seedsB, label + ": seeds of B");
        checkEqual(Row(row.begin() + 11, row.end()),
                   {std::to_string(outcome.spreadA), std::to_string(outcome.spreadB), std::to_string(outcome.neutral),
                    std::to_string(outcome.allocation.rankingOverlap), std::to_string(outcome.allocation.skippedA),
                    std::to_string(outcome.allocation.skippedB), std::to_string(outcome.spread.rounds),
                    engine::stopReasonName(outcome.spread.stopReason)},
                   label + ": outcome");
    }
}

// A player's seed is the same against every opponent and under every sequence. Each of the 30
// (player, algorithm, run) seeds is recorded against all five opponents under both sequences.
void seedsPerPlayer() {
    const Fixture fixture;
    const auto results = readRows(fixture.run(fixture.config(), "batch").first);
    std::map<std::string, std::string> seedOf;
    std::map<std::string, std::set<std::string>> opponents;
    std::map<std::string, std::set<std::string>> sequences;
    for (const auto& row : results) {
        for (const bool playerA : {true, false}) {
            const auto& own = playerA ? row[3] : row[4];
            const auto& seed = playerA ? row[7] : row[8];
            const auto key = std::string(playerA ? "A " : "B ") + own + ", budget " + row[5] + ", run " + row[6];
            const auto [entry, inserted] = seedOf.emplace(key, seed);
            check(inserted || entry->second == seed, key + ": the seed changed with the opponent or the sequence");
            opponents[key].insert(playerA ? row[4] : row[3]);
            sequences[key].insert(row[9]);
        }
    }
    check(seedOf.size() == 30, "5 algorithms x 3 runs x 2 players");
    for (const auto& [key, seed] : seedOf) {
        check(opponents[key].size() == 5, key + ": recorded against all five opponents");
        check(sequences[key].size() == 2, key + ": recorded under both sequences");
    }
}

// The same graph file in another folder under another name, played at another threshold, gives
// the same seeds.
void seedsIgnoreThresholdAndLocation() {
    const Fixture fixture;
    const auto config = fixture.config();
    const auto results = readRows(fixture.run(config, "batch").first);
    std::filesystem::create_directory(fixture.scratch / "elsewhere");
    const auto copy = fixture.scratch / "elsewhere" / "copy.mtx";
    std::filesystem::copy_file(fixture.graph, copy);
    auto moved = config;
    moved.graph = copy.string();
    moved.theta = 0.75;
    const auto movedResults = readRows(fixture.run(moved, "moved").first);
    check(movedResults.size() == results.size(), "same number of matches");
    for (size_t index = 0; index < results.size(); ++index) {
        check(movedResults[index][0] == "copy.mtx" && movedResults[index][10] == "0.75", "the moved batch was run");
        check(movedResults[index][7] == results[index][7] && movedResults[index][8] == results[index][8],
              "row " + std::to_string(index + 1) + ": the seeds changed with the threshold or the file's location");
    }
}

// Checks that each row of `reduced` is the row of `full` with the same match. The match is given
// by the columns `a` (algorithm_a, followed by algorithm_b), `run` and `sequence`.
void checkKeptRows(const std::filesystem::path& full, const std::filesystem::path& reduced, size_t a, size_t run,
                   size_t sequence, size_t expectedRows, const std::string& label) {
    const auto key = [&](const Row& row) { return row[a] + " " + row[a + 1] + " " + row[sequence] + " " + row[run]; };
    std::map<std::string, Row> fullRows;
    for (const auto& row : readRows(full)) fullRows[key(row)] = row;
    const auto rows = readRows(reduced);
    check(rows.size() == expectedRows, label + ": one row per planned match");
    for (const auto& row : rows) {
        check(fullRows.count(key(row)) == 1, label + ", " + key(row) + ": not in the full batch");
        checkEqual(row, fullRows[key(row)], label + ", " + key(row));
    }
}

// With deterministic_runs = 1 the pairs without Random play run 0 only; every row it keeps is the
// row of the full batch, seeds included.
void reducedPlan() {
    const Fixture fixture;
    auto config = fixture.config();
    const auto full = fixture.run(config, "full");
    config.deterministicRuns = 1;
    const auto reduced = fixture.run(config, "reduced");
    const size_t rows = engine::plannedMatches(config).size();
    check(rows == (9 * 3 + 16 * 1) * 2, "(9 pairs with Random x 3 runs + 16 pairs x 1 run) x 2 sequences");
    checkKeptRows(full.first, reduced.first, 3, 6, 9, rows, "results");
    checkKeptRows(full.second, reduced.second, 1, 4, 7, rows, "trace");
}

void repeatedBatch() {
    const Fixture fixture;
    const auto first = fixture.run(fixture.config(), "first");
    const auto second = fixture.run(fixture.config(), "second");
    check(readFile(first.first) == readFile(second.first), "results differ between two runs");
    check(readFile(first.second) == readFile(second.second), "traces differ between two runs");
}

// The batch plays whatever ordering the ranking file holds, here the nodes by decreasing ID.
void storedRankingIsPlayed() {
    const Fixture fixture;
    auto config = fixture.config();
    config.algorithms = {"degree", "betweenness"};
    config.betweennessRanking = writeFile(fixture.scratch / "reversed.u32le",
                                          littleEndian({11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0})).string();
    for (const auto& row : readRows(fixture.run(config, "reversed").second)) {
        if (row[1] == "betweenness") checkEqual(parseList(row[9]), {11, 10, 9, 8, 7, 6}, "candidates of A");
        if (row[2] == "betweenness") checkEqual(parseList(row[10]), {11, 10, 9, 8, 7, 6}, "candidates of B");
    }
}

void rejectedBatches() {
    const Fixture fixture;
    const auto output = (fixture.scratch / "rejected.csv").string();
    auto tooLarge = fixture.config();
    tooLarge.budget = 7;
    rejects([&] { engine::runBatch(tooLarge, output, output); }, "2k above the node count", "twice the budget");
    auto shortRanking = fixture.config();
    shortRanking.betweennessRanking = writeFile(fixture.scratch / "short.u32le", littleEndian({0, 1, 2})).string();
    rejects([&] { engine::runBatch(shortRanking, output, output); }, "ranking of another size", "one entry per node");
    auto repeated = fixture.config();
    repeated.betweennessRanking = writeFile(fixture.scratch / "repeated.u32le",
                                            littleEndian({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10})).string();
    rejects([&] { engine::runBatch(repeated, output, output); }, "ranking with a repeated node", "not an ordering");
    auto missing = fixture.config();
    missing.graph = (fixture.scratch / "missing.mtx").string();
    rejects([&] { engine::runBatch(missing, output, output); }, "missing graph", "Cannot open");
}

}  // namespace

int main() {
    return runTests({
        {"config keys", configKeys},
        {"config values", configValues},
        {"config file", configFile},
        {"planned matches", plannedMatches},
        {"csv fields", csvFields},
        {"batch files", batchFiles},
        {"seeds per player", seedsPerPlayer},
        {"seeds ignore threshold and location", seedsIgnoreThresholdAndLocation},
        {"reduced plan", reducedPlan},
        {"repeated batch", repeatedBatch},
        {"stored ranking is played", storedRankingIsPlayed},
        {"rejected batches", rejectedBatches},
    });
}
