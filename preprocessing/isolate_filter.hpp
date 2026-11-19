#pragma once

// Removes the nodes without edges from a graph, as the study did for every network. A node is
// removed when it has no neighbour in the engine's undirected view (self-loops do not count).
// The remaining nodes keep their order and are renumbered 0..n'-1. The output folder receives
//   filtered.mtx   the graph without those nodes (only written when some were removed)
//   old_new.tsv    old and new ID of every remaining node (0-based)
//   removed.tsv    old ID of every removed node (0-based)
//   report.json    node counts before and after

#include "engine/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace isolate_filter {

namespace fs = std::filesystem;
inline constexpr uint32_t removedId = std::numeric_limits<uint32_t>::max();

struct Result {
    std::string graphPath;           // filtered.mtx, or the input when nothing was removed
    uint32_t nodesBefore = 0;
    uint64_t undirectedEdges = 0;
    std::vector<uint32_t> oldToNew;  // removedId for a removed node
    std::vector<uint32_t> newToOld;
    std::vector<uint32_t> removed;

    bool changed() const { return !removed.empty(); }
};

inline Result renumber(const engine::Graph& graph, const std::string& input) {
    Result result;
    result.graphPath = input;
    result.nodesBefore = graph.nodeCount();
    result.undirectedEdges = graph.edgeCount();
    result.oldToNew.assign(graph.nodeCount(), removedId);
    for (uint32_t old = 0; old < graph.nodeCount(); ++old) {
        if (graph.degree(old) == 0) {
            result.removed.push_back(old);
        } else {
            result.oldToNew[old] = static_cast<uint32_t>(result.newToOld.size());
            result.newToOld.push_back(old);
        }
    }
    if (result.newToOld.empty()) throw std::runtime_error("every node is isolated; the filtered graph would be empty");
    return result;
}

// One line per undirected edge (larger new ID, smaller new ID), in increasing old-ID order.
inline void writeMatrix(const engine::Graph& graph, const Result& result) {
    std::ofstream output(result.graphPath);
    output << "%%MatrixMarket matrix coordinate pattern symmetric\n"
              "% Isolates removed after the engine's unique loop-free undirected symmetrization.\n";
    output << result.newToOld.size() << ' ' << result.newToOld.size() << ' ' << graph.edgeCount() << '\n';
    for (const auto old : result.newToOld) {
        for (uint32_t neighbor : graph.neighbors(old)) {
            if (neighbor < old) output << result.oldToNew[old] + 1 << ' ' << result.oldToNew[neighbor] + 1 << '\n';
        }
    }
    if (!output.flush()) throw std::runtime_error("cannot write " + result.graphPath);
}

// The filtered graph must have exactly the old neighbourhoods, renumbered.
inline void checkNeighbourhoods(const engine::Graph& original, const engine::Graph& filtered, const Result& result) {
    if (filtered.nodeCount() != result.newToOld.size() || filtered.edgeCount() != original.edgeCount()) {
        throw std::runtime_error("the filtered graph has other node or edge counts than expected");
    }
    for (uint32_t id = 0; id < filtered.nodeCount(); ++id) {
        const auto old = result.newToOld[id];
        std::vector<uint32_t> expected;
        for (uint32_t neighbor : original.neighbors(old)) expected.push_back(result.oldToNew[neighbor]);
        const auto actual = filtered.neighbors(id);
        if (!std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) throw std::runtime_error("a neighbourhood changed in the filtered graph");
    }
}

inline void writeMappings(const fs::path& output, const Result& result) {
    std::ofstream mapping(output / "old_new.tsv");
    mapping << "old_internal_id_0based\tnew_internal_id_0based\n";
    for (uint32_t id = 0; id < result.newToOld.size(); ++id) mapping << result.newToOld[id] << '\t' << id << '\n';
    std::ofstream removed(output / "removed.tsv");
    removed << "old_internal_id_0based\n";
    for (const auto old : result.removed) removed << old << '\n';
    if (!mapping.flush() || !removed.flush()) throw std::runtime_error("cannot write the ID mappings");
}

inline std::string reportJson(const Result& result) {
    return "{\"graph\":\"" + result.graphPath + "\""
        + ",\"nodes_before\":" + std::to_string(result.nodesBefore)
        + ",\"isolated_nodes_removed\":" + std::to_string(result.removed.size())
        + ",\"nodes_after\":" + std::to_string(result.newToOld.size())
        + ",\"undirected_edges\":" + std::to_string(result.undirectedEdges) + "}\n";
}

// Filters `input` into the new folder `output`.
inline Result run(const std::string& input, const fs::path& output) {
    const auto original = engine::loadGraph(input);
    auto result = renumber(original, input);
    if (!fs::create_directory(output)) throw std::runtime_error("output folder already exists: " + output.string());
    if (result.changed()) {
        result.graphPath = (output / "filtered.mtx").string();
        writeMatrix(original, result);
        checkNeighbourhoods(original, engine::loadGraph(result.graphPath), result);
    }
    writeMappings(output, result);
    std::ofstream(output / "report.json") << reportJson(result);
    return result;
}

}  // namespace isolate_filter
