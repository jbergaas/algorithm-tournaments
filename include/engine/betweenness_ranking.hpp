#pragma once

// Stored Betweenness of one graph, written by `engine --betweenness-ranking`:
//   scores.f64le       the n scores as little-endian IEEE 754 doubles, in node order
//   fullranking.u32le  all n nodes by decreasing score (ties by ID) as little-endian uint32
// A batch plays the first 2k nodes of fullranking.u32le as Betweenness's candidates.

#include "algorithms.hpp"
#include "graph.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace engine {

template <class T>
void writeLittleEndian(const std::filesystem::path& path, const std::vector<T>& values) {
    std::string bytes;
    bytes.reserve(values.size() * sizeof(T));
    for (const T value : values) {
        uint64_t bits = 0;
        if constexpr (std::is_same_v<T, double>) bits = std::bit_cast<uint64_t>(value);
        else bits = value;
        for (unsigned byte = 0; byte < sizeof(T); ++byte) bytes.push_back(static_cast<char>(bits >> (8 * byte)));
    }
    std::ofstream output(path, std::ios::binary);
    if (!output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

inline void writeBetweennessRanking(const Graph& graph, int threads, const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const auto scores = betweennessScores(graph, threads);
    writeLittleEndian(directory / "scores.f64le", scores);
    writeLittleEndian(directory / "fullranking.u32le", rankByScore(scores));
}

// Reads a fullranking.u32le file and checks that it lists every node of the graph once.
inline std::vector<uint32_t> loadBetweennessRanking(const std::string& path, uint32_t nodeCount) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open Betweenness ranking: " + path);
    const std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (bytes.size() != 4 * static_cast<uint64_t>(nodeCount)) {
        throw std::runtime_error("Betweenness ranking " + path + " does not have one entry per node");
    }
    std::vector<uint32_t> ranking(nodeCount);
    std::vector<bool> seen(nodeCount, false);
    for (uint32_t index = 0; index < nodeCount; ++index) {
        uint32_t node = 0;
        for (unsigned byte = 0; byte < 4; ++byte) {
            node |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[4 * index + byte])) << (8 * byte);
        }
        if (node >= nodeCount || seen[node]) {
            throw std::runtime_error("Betweenness ranking " + path + " is not an ordering of the nodes");
        }
        seen[node] = true;
        ranking[index] = node;
    }
    return ranking;
}

}  // namespace engine
