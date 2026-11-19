#pragma once

// Competitive linear-threshold spread. In each round every neutral node with an owned neighbour
// is checked: it joins the player that owns strictly more of its neighbours, provided that
// player owns at least theta times the node's degree of them. All changes of a round happen at
// once, and an owned node never changes again. The spread stops when a round changes nothing.

#include "graph.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace engine {

enum class Owner : uint8_t { Neutral = 0, A = 1, B = 2 };

// empty_frontier: no neutral node is next to a seed or to a node won in the last round;
// no_acquisitions: some were checked, but none met the rule.
enum class StopReason { EmptyFrontier, NoAcquisitions };

inline const char* stopReasonName(StopReason reason) {
    return reason == StopReason::EmptyFrontier ? "empty_frontier" : "no_acquisitions";
}

struct SpreadResult {
    uint32_t rounds = 0;   // rounds in which at least one node changed owner
    StopReason stopReason = StopReason::EmptyFrontier;
};

// Runs the spread on `ownership` (one entry per node) until it stops.
inline SpreadResult spread(const Graph& graph, std::vector<Owner>& ownership, double theta) {
    const uint32_t n = graph.nodeCount();
    std::vector<uint32_t> countA(n, 0);
    std::vector<uint32_t> countB(n, 0);
    std::vector<uint8_t> inFrontier(n, 0);
    std::vector<uint32_t> frontier;
    std::vector<uint32_t> nextFrontier;
    std::vector<std::pair<uint32_t, Owner>> changes;

    // Counts one more neighbour owned by `owner` at a neutral node and puts it on the frontier.
    auto bump = [&](uint32_t node, Owner owner, std::vector<uint32_t>& bucket) {
        if (ownership[node] != Owner::Neutral) return;
        if (owner == Owner::A) ++countA[node];
        else ++countB[node];
        if (!inFrontier[node]) {
            inFrontier[node] = 1;
            bucket.push_back(node);
        }
    };
    auto meetsThreshold = [theta](uint32_t owned, uint32_t degree) {
        return static_cast<double>(owned) >= theta * static_cast<double>(degree);
    };

    for (uint32_t node = 0; node < n; ++node) {
        if (ownership[node] == Owner::Neutral) continue;
        for (uint32_t neighbor : graph.neighbors(node)) bump(neighbor, ownership[node], frontier);
    }

    for (uint32_t rounds = 0;; ++rounds) {
        if (frontier.empty()) return {rounds, StopReason::EmptyFrontier};

        changes.clear();
        for (uint32_t node : frontier) {
            inFrontier[node] = 0;
            if (ownership[node] != Owner::Neutral) continue;
            const uint32_t degree = graph.degree(node);
            if (countA[node] > countB[node] && meetsThreshold(countA[node], degree)) {
                changes.push_back({node, Owner::A});
            } else if (countB[node] > countA[node] && meetsThreshold(countB[node], degree)) {
                changes.push_back({node, Owner::B});
            }
        }
        frontier.clear();
        if (changes.empty()) return {rounds, StopReason::NoAcquisitions};

        for (const auto& [node, owner] : changes) ownership[node] = owner;
        nextFrontier.clear();
        for (const auto& [node, owner] : changes) {
            for (uint32_t neighbor : graph.neighbors(node)) bump(neighbor, owner, nextFrontier);
        }
        frontier.swap(nextFrontier);
    }
}

}  // namespace engine
