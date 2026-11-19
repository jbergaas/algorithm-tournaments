#pragma once

// One match: both players' 2k candidates go through the picking sequence, which hands out k
// seeds per player, and the seeds then spread (spread.hpp).

#include "graph.hpp"
#include "spread.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine {

// sequential_a: A takes all k seeds, then B; sequential_b: the other way round;
// alternating_a: A, B, A, B, ...; snake: A, B, B, A, A, B, B, A, ...
enum class Sequence { SequentialA, SequentialB, AlternatingA, Snake };

inline const char* sequenceName(Sequence sequence) {
    switch (sequence) {
        case Sequence::SequentialA: return "sequential_a";
        case Sequence::SequentialB: return "sequential_b";
        case Sequence::AlternatingA: return "alternating_a";
        case Sequence::Snake: return "snake";
    }
    return "";
}

inline Sequence parseSequence(const std::string& name) {
    for (auto sequence : {Sequence::SequentialA, Sequence::SequentialB, Sequence::AlternatingA, Sequence::Snake}) {
        if (name == sequenceName(sequence)) return sequence;
    }
    throw std::invalid_argument("unknown sequence: " + name);
}

// The player who picks at each of the 2k turns.
inline std::vector<Owner> pickOrder(Sequence sequence, uint32_t k) {
    std::vector<Owner> order;
    for (uint32_t turn = 0; turn < 2 * k; ++turn) {
        bool playerA = false;
        switch (sequence) {
            case Sequence::SequentialA: playerA = turn < k; break;
            case Sequence::SequentialB: playerA = turn >= k; break;
            case Sequence::AlternatingA: playerA = turn % 2 == 0; break;
            case Sequence::Snake: playerA = turn % 4 == 0 || turn % 4 == 3; break;
        }
        order.push_back(playerA ? Owner::A : Owner::B);
    }
    return order;
}

struct Allocation {
    std::vector<Owner> ownership;   // one entry per node: the seeds, everything else neutral
    std::vector<uint32_t> seedsA;
    std::vector<uint32_t> seedsB;
    uint32_t rankingOverlap = 0;    // nodes on both candidate lists
    uint32_t skippedA = 0;          // A's candidates passed over because B had taken them
    uint32_t skippedB = 0;
};

// Each player takes, at its turns, its best candidate that nobody holds yet. Both lists hold 2k
// distinct nodes, so neither runs out.
inline Allocation allocateSeeds(uint32_t nodeCount, const std::vector<uint32_t>& rankingA,
                                const std::vector<uint32_t>& rankingB, uint32_t k, Sequence sequence) {
    Allocation result;
    result.ownership.assign(nodeCount, Owner::Neutral);
    std::vector<uint32_t> sortedA(rankingA);
    std::sort(sortedA.begin(), sortedA.end());
    for (uint32_t node : rankingB) {
        result.rankingOverlap += std::binary_search(sortedA.begin(), sortedA.end(), node);
    }
    size_t cursorA = 0;
    size_t cursorB = 0;
    for (Owner player : pickOrder(sequence, k)) {
        const bool isA = player == Owner::A;
        const auto& ranking = isA ? rankingA : rankingB;
        size_t& cursor = isA ? cursorA : cursorB;
        uint32_t& skipped = isA ? result.skippedA : result.skippedB;
        while (result.ownership[ranking[cursor]] != Owner::Neutral) {
            ++skipped;
            ++cursor;
        }
        const uint32_t node = ranking[cursor++];
        result.ownership[node] = player;
        (isA ? result.seedsA : result.seedsB).push_back(node);
    }
    return result;
}

struct MatchResult {
    Allocation allocation;
    SpreadResult spread;
    uint32_t spreadA = 0;   // nodes A owns at the end, its seeds included
    uint32_t spreadB = 0;
    uint32_t neutral = 0;
};

inline MatchResult playMatch(const Graph& graph, const std::vector<uint32_t>& rankingA,
                             const std::vector<uint32_t>& rankingB, uint32_t k, Sequence sequence,
                             double theta) {
    MatchResult result;
    result.allocation = allocateSeeds(graph.nodeCount(), rankingA, rankingB, k, sequence);
    std::vector<Owner> ownership = std::move(result.allocation.ownership);
    result.spread = spread(graph, ownership, theta);
    for (Owner owner : ownership) {
        if (owner == Owner::A) ++result.spreadA;
        else if (owner == Owner::B) ++result.spreadB;
        else ++result.neutral;
    }
    return result;
}

}  // namespace engine
