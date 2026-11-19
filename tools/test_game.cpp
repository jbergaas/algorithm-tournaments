// Matches (game.hpp): the picking sequences, the seeds each player gets from its candidate list
// with the overlap and skip counts, and that a match spreads at the threshold it is given.

#include "check.hpp"
#include "engine/game.hpp"
#include "engine/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace {

using engine::Owner;
using engine::Sequence;
using Nodes = std::vector<uint32_t>;

constexpr Owner A = Owner::A;
constexpr Owner B = Owner::B;

void sequences() {
    checkEqual(engine::pickOrder(Sequence::SequentialA, 2), {A, A, B, B}, "sequential_a");
    checkEqual(engine::pickOrder(Sequence::SequentialB, 2), {B, B, A, A}, "sequential_b");
    checkEqual(engine::pickOrder(Sequence::AlternatingA, 2), {A, B, A, B}, "alternating_a");
    checkEqual(engine::pickOrder(Sequence::Snake, 4), {A, B, B, A, A, B, B, A}, "snake");
    check(engine::pickOrder(Sequence::Snake, 0).empty(), "no turns for k = 0");
    for (auto sequence : {Sequence::SequentialA, Sequence::SequentialB, Sequence::AlternatingA, Sequence::Snake}) {
        check(engine::parseSequence(engine::sequenceName(sequence)) == sequence,
              std::string("name of ") + engine::sequenceName(sequence));
    }
    rejects([] { engine::parseSequence("unknown"); }, "unknown sequence name", "unknown sequence");
}

// Checks the seeds, the overlap and skip counts, and that the ownership holds exactly the k
// distinct seeds of each player.
void checkAllocation(const Nodes& rankingA, const Nodes& rankingB, uint32_t k, Sequence sequence, uint32_t nodeCount,
                     const Nodes& seedsA, const Nodes& seedsB, uint32_t overlap, uint32_t skippedA, uint32_t skippedB,
                     const std::string& label) {
    const auto allocation = engine::allocateSeeds(nodeCount, rankingA, rankingB, k, sequence);
    checkEqual(allocation.seedsA, seedsA, label + ": seeds of A");
    checkEqual(allocation.seedsB, seedsB, label + ": seeds of B");
    check(allocation.rankingOverlap == overlap, label + ": overlap " + std::to_string(allocation.rankingOverlap));
    check(allocation.skippedA == skippedA, label + ": skips of A " + std::to_string(allocation.skippedA));
    check(allocation.skippedB == skippedB, label + ": skips of B " + std::to_string(allocation.skippedB));
    check(allocation.ownership.size() == nodeCount, label + ": one owner per node");
    std::set<uint32_t> distinct(seedsA.begin(), seedsA.end());
    distinct.insert(seedsB.begin(), seedsB.end());
    check(distinct.size() == 2 * k, label + ": 2k distinct seeds");
    for (uint32_t node = 0; node < nodeCount; ++node) {
        const bool isA = std::count(seedsA.begin(), seedsA.end(), node) > 0;
        const bool isB = std::count(seedsB.begin(), seedsB.end(), node) > 0;
        const Owner owner = isA ? A : isB ? B : Owner::Neutral;
        check(allocation.ownership[node] == owner, label + ": owner of node " + std::to_string(node));
    }
}

void identicalRankings() {
    const Nodes ranking = {1, 2, 3, 4, 5, 6};
    checkAllocation(ranking, ranking, 3, Sequence::SequentialA, 7, {1, 2, 3}, {4, 5, 6}, 6, 0, 3, "sequential_a");
    checkAllocation(ranking, ranking, 3, Sequence::SequentialB, 7, {4, 5, 6}, {1, 2, 3}, 6, 3, 0, "sequential_b");
    checkAllocation(ranking, ranking, 3, Sequence::AlternatingA, 7, {1, 3, 5}, {2, 4, 6}, 6, 2, 3, "alternating_a");
    checkAllocation(ranking, ranking, 3, Sequence::Snake, 7, {1, 4, 5}, {2, 3, 6}, 6, 2, 3, "snake");
}

void overlaps() {
    checkAllocation({1, 2, 3, 4, 5, 6}, {1, 3, 7, 8, 9, 10}, 3, Sequence::Snake, 11, {1, 2, 4}, {3, 7, 8}, 2, 1, 1,
                    "partial overlap");
    checkAllocation({0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}, 3, Sequence::Snake, 12, {0, 1, 2}, {6, 7, 8}, 0, 0, 0,
                    "no overlap");
    checkAllocation({0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, 3, Sequence::Snake, 6, {0, 1, 2}, {5, 4, 3}, 6, 0, 0,
                    "complete overlap without skips");
    checkAllocation({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}, 2, Sequence::Snake, 6, {0, 3}, {1, 2}, 4, 2, 0,
                    "rankings longer than 2k");
}

void smallAndOddBudgets() {
    const Nodes two = {0, 1};
    checkAllocation(two, two, 1, Sequence::Snake, 2, {0}, {1}, 2, 0, 1, "snake, k = 1");
    checkAllocation(two, two, 1, Sequence::SequentialB, 2, {1}, {0}, 2, 1, 0, "sequential_b, k = 1");
    const Nodes four = {0, 1, 2, 3};
    checkAllocation(four, four, 2, Sequence::SequentialA, 4, {0, 1}, {2, 3}, 4, 0, 2, "sequential_a, k = 2");
    checkAllocation(four, four, 2, Sequence::SequentialB, 4, {2, 3}, {0, 1}, 4, 2, 0, "sequential_b, k = 2");
    checkAllocation(four, four, 2, Sequence::AlternatingA, 4, {0, 2}, {1, 3}, 4, 1, 2, "alternating_a, k = 2");
    checkAllocation(four, four, 2, Sequence::Snake, 4, {0, 3}, {1, 2}, 4, 2, 1, "snake, k = 2");
    const Nodes six = {0, 1, 2, 3, 4, 5};
    checkAllocation(six, six, 3, Sequence::Snake, 6, {0, 3, 4}, {1, 2, 5}, 6, 2, 3, "snake, k = 3");
    const Nodes eight = {0, 1, 2, 3, 4, 5, 6, 7};
    checkAllocation(eight, eight, 4, Sequence::Snake, 8, {0, 3, 4, 7}, {1, 2, 5, 6}, 8, 4, 3, "snake, k = 4");
}

void matches() {
    // A star with centre 0 and leaves 1..10, and node 11 apart. A gets leaf 1, B node 11.
    engine::Graph::Builder builder(12);
    for (uint32_t leaf = 1; leaf <= 10; ++leaf) builder.addEdge(0, leaf);
    const auto star = builder.build();
    const auto strict = engine::playMatch(star, {1, 2}, {11, 3}, 1, Sequence::AlternatingA, 0.5);
    check(strict.spreadA == 1 && strict.spreadB == 1 && strict.neutral == 10, "theta 0.5 reaches the spread");
    const auto loose = engine::playMatch(star, {1, 2}, {11, 3}, 1, Sequence::AlternatingA, 0.0);
    check(loose.spreadA == 11 && loose.spreadB == 1 && loose.neutral == 0, "theta 0 reaches the spread");
    check(loose.spread.rounds == 2 && loose.spread.stopReason == engine::StopReason::EmptyFrontier,
          "rounds and stop reason of the match");

    // A takes its first candidate, 2, before B's turn, so B gets 1: the lists' order counts, not the IDs.
    const auto path = engine::Graph::Builder(3).addEdge(0, 2).build();
    const auto ordered = engine::playMatch(path, {2, 1}, {1, 2}, 1, Sequence::AlternatingA, 0.0);
    checkEqual(ordered.allocation.seedsA, {2}, "seeds of A follow A's list");
    checkEqual(ordered.allocation.seedsB, {1}, "seeds of B follow B's list");
    check(ordered.spreadA == 2 && ordered.spreadB == 1 && ordered.neutral == 0, "spread of the ordered match");
    check(ordered.allocation.rankingOverlap == 2 && ordered.allocation.skippedA == 0
              && ordered.allocation.skippedB == 0,
          "overlap without skips");
}

}  // namespace

int main() {
    return runTests({
        {"sequences", sequences},
        {"identical rankings", identicalRankings},
        {"overlaps", overlaps},
        {"small and odd budgets", smallAndOddBudgets},
        {"matches", matches},
    });
}
