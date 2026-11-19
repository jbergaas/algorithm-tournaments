#pragma once

// The graph the game is played on: the simple undirected view of a Matrix Market file. Every
// entry (i, j) with i != j links i and j; direction, values, repeated entries and self-loops are
// dropped. Each node's neighbours are stored in increasing ID order.

#include <fast_matrix_market/fast_matrix_market.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

class Graph {
    uint32_t nodeCount_ = 0;
    uint32_t entryCount_ = 0;
    std::vector<uint32_t> offsets_{0};
    std::vector<uint32_t> neighbors_;

public:
    uint32_t nodeCount() const { return nodeCount_; }
    // Entries as read, with a symmetric file expanded to both directions, before the undirected view.
    uint32_t entryCount() const { return entryCount_; }
    // Edges of the undirected view.
    uint64_t edgeCount() const { return neighbors_.size() / 2; }

    std::span<const uint32_t> neighbors(uint32_t node) const {
        return {neighbors_.data() + offsets_[node], neighbors_.data() + offsets_[node + 1]};
    }

    uint32_t degree(uint32_t node) const { return offsets_[node + 1] - offsets_[node]; }

    class Builder {
        std::vector<std::vector<uint32_t>> adjacency_;
        uint32_t entryCount_ = 0;

    public:
        explicit Builder(uint32_t nodeCount) : adjacency_(nodeCount) {}

        // Adds the entry (from, to).
        Builder& addEdge(uint32_t from, uint32_t to) {
            if (from >= adjacency_.size() || to >= adjacency_.size()) {
                throw std::out_of_range("node index out of range");
            }
            ++entryCount_;
            if (from != to) {
                adjacency_[from].push_back(to);
                adjacency_[to].push_back(from);
            }
            return *this;
        }

        Graph build() {
            Graph graph;
            graph.nodeCount_ = static_cast<uint32_t>(adjacency_.size());
            graph.entryCount_ = entryCount_;
            graph.offsets_.reserve(adjacency_.size() + 1);
            for (auto& neighbors : adjacency_) {
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                graph.neighbors_.insert(graph.neighbors_.end(), neighbors.begin(), neighbors.end());
                graph.offsets_.push_back(static_cast<uint32_t>(graph.neighbors_.size()));
                std::vector<uint32_t>().swap(neighbors);
            }
            return graph;
        }
    };
};

inline Graph loadGraph(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    fast_matrix_market::matrix_market_header header;
    std::vector<int64_t> rows, columns;
    std::vector<double> values;
    // A symmetric file comes back with both entries of every off-diagonal edge.
    fast_matrix_market::read_matrix_market_triplet(file, header, rows, columns, values);
    Graph::Builder builder(static_cast<uint32_t>(std::max(header.nrows, header.ncols)));
    for (size_t i = 0; i < rows.size(); ++i) {
        builder.addEdge(static_cast<uint32_t>(rows[i]), static_cast<uint32_t>(columns[i]));
    }
    return builder.build();
}

}  // namespace engine
