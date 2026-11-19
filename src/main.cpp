// The engine.
//   engine BATCH.cfg --output results.csv --trace rankings.csv
//       plays every match of a batch (batch.hpp) and writes one row per match to each file;
//   engine --betweenness-ranking GRAPH.mtx --threads N --output DIR
//       computes the exact Betweenness of a graph and writes DIR/scores.f64le and
//       DIR/fullranking.u32le (betweenness_ranking.hpp).

#include "engine/batch.hpp"
#include "engine/betweenness_ranking.hpp"
#include "engine/graph.hpp"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

const char* const kUsage =
    "Usage:\n"
    "  engine BATCH.cfg --output results.csv --trace rankings.csv\n"
    "  engine --betweenness-ranking GRAPH.mtx --threads N --output DIR\n";

// Reads "--name value" pairs from argv[first..] and checks that exactly `names` are given.
std::map<std::string, std::string> options(int argc, char* argv[], int first, std::initializer_list<const char*> names) {
    std::map<std::string, std::string> values;
    for (int index = first; index + 1 < argc; index += 2) values[argv[index]] = argv[index + 1];
    if ((argc - first) % 2 != 0 || values.size() != names.size()) throw std::invalid_argument(kUsage);
    for (const char* name : names) {
        if (!values.count(name)) throw std::invalid_argument(kUsage);
    }
    return values;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) throw std::invalid_argument(kUsage);
        if (std::string(argv[1]) == "--betweenness-ranking") {
            if (argc < 3) throw std::invalid_argument(kUsage);
            const auto values = options(argc, argv, 3, {"--threads", "--output"});
            const int threads = std::stoi(values.at("--threads"));
            if (threads < 1) throw std::invalid_argument("--threads must be at least 1");
            engine::writeBetweennessRanking(engine::loadGraph(argv[2]), threads, values.at("--output"));
            return 0;
        }
        const auto values = options(argc, argv, 2, {"--output", "--trace"});
        const auto matches = engine::runBatch(engine::loadBatchConfig(argv[1]), values.at("--output"), values.at("--trace"));
        std::cout << matches << " matches written to " << values.at("--output") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
