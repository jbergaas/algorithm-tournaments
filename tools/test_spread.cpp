// The competitive threshold spread (spread.hpp): the threshold rule with its ties and boundaries,
// round counts and stop reasons, synchronous and irreversible updates, the undirected view, and
// a plain round-by-round reference compared on random and real graphs.

#include "check.hpp"
#include "engine/graph.hpp"
#include "engine/spread.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::Graph;
using engine::Owner;
using engine::SpreadResult;
using engine::StopReason;
using Entries = std::vector<std::pair<uint32_t, uint32_t>>;

constexpr Owner N = Owner::Neutral;
constexpr Owner A = Owner::A;
constexpr Owner B = Owner::B;

Graph makeGraph(uint32_t nodeCount, const Entries& entries) {
    Graph::Builder builder(nodeCount);
    for (const auto& [from, to] : entries) builder.addEdge(from, to);
    return builder.build();
}

// Runs the spread and checks the final ownership, the rounds and the stop reason.
void checkSpread(const Graph& graph, std::vector<Owner> ownership, double theta, const std::vector<Owner>& expected,
                 uint32_t rounds, StopReason stopReason, const std::string& label) {
    const auto result = engine::spread(graph, ownership, theta);
    checkEqual(ownership, expected, label + ": ownership");
    check(result.rounds == rounds, label + ": " + std::to_string(result.rounds) + " rounds, expected "
                                       + std::to_string(rounds));
    check(result.stopReason == stopReason, label + ": stopped with " + engine::stopReasonName(result.stopReason));
}

// Node 0 with `a` neighbours owned by A, `b` owned by B and `neutral` neutral ones (entries in
// both directions); returns node 0's owner after the spread.
Owner centreAfterSpread(uint32_t a, uint32_t b, uint32_t neutral, double theta) {
    const uint32_t degree = a + b + neutral;
    Graph::Builder builder(degree + 1);
    std::vector<Owner> ownership(degree + 1, N);
    for (uint32_t leaf = 1; leaf <= degree; ++leaf) {
        if (leaf % 2 == 0) builder.addEdge(0, leaf);
        else builder.addEdge(leaf, 0);
        if (leaf <= a) ownership[leaf] = A;
        else if (leaf <= a + b) ownership[leaf] = B;
    }
    engine::spread(builder.build(), ownership, theta);
    return ownership[0];
}

void thresholdRule() {
    struct Case {
        uint32_t a;
        uint32_t b;
        uint32_t neutral;
        double theta;
        Owner centre;
        const char* label;
    };
    const std::vector<Case> cases = {
        {1, 0, 9, 0.0, A, "theta 0: a strict majority of the owned neighbours suffices"},
        {1, 2, 7, 0.0, B, "theta 0: the rule is the same for B"},
        {1, 0, 9, 0.5, N, "1 A, 0 B, 9 neutral fail theta 0.5"},
        {5, 4, 1, 0.5, A, "5 A, 4 B, 1 neutral meet theta 0.5 exactly"},
        {5, 5, 0, 0.5, N, "5 A and 5 B tie"},
        {6, 2, 2, 0.5, A, "6 A, 2 B, 2 neutral meet theta 0.5"},
        {4, 5, 1, 0.5, B, "4 A, 5 B, 1 neutral: B meets theta 0.5"},
        {1, 0, 9, 0.1, A, "1 of 10 meets theta 0.1"},
        {1, 0, 9, std::nextafter(0.1, 1.0), N, "1 of 10 fails the next double above 0.1"},
        {1, 0, 1, 0.0, A, "1 A, 1 neutral at theta 0"},
        {2, 0, 0, 1.0, A, "all neighbours meet theta 1"},
        {1, 0, 1, 1.0, N, "half of the neighbours fail theta 1"},
    };
    for (const auto& c : cases) check(centreAfterSpread(c.a, c.b, c.neutral, c.theta) == c.centre, c.label);
}

void roundsAndStopReasons() {
    const auto path = makeGraph(3, {{0, 1}, {1, 2}});
    checkSpread(path, {N, N, N}, 0.0, {N, N, N}, 0, StopReason::EmptyFrontier, "no seeds");
    checkSpread(path, {A, N, B}, 0.0, {A, N, B}, 0, StopReason::NoAcquisitions, "tie at the only checked node");
    checkSpread(path, {A, N, N}, 0.0, {A, A, A}, 2, StopReason::EmptyFrontier, "chain");

    // Node 2 has degree 3, so after node 1 joins A it has 1 of 3, below theta 0.5.
    const auto fork = makeGraph(5, {{0, 1}, {1, 2}, {2, 3}, {2, 4}});
    checkSpread(fork, {A, N, N, N, N}, 0.5, {A, A, N, N, N}, 1, StopReason::NoAcquisitions, "fork");

    const auto isolated = makeGraph(2, {});
    checkSpread(isolated, {N, A}, 0.0, {N, A}, 0, StopReason::EmptyFrontier, "isolated node");
}

// A directed chain 0 -> 1 -> 2 spreads both ways.
void directedChain() {
    const auto chain = makeGraph(3, {{0, 1}, {1, 2}});
    checkSpread(chain, {A, N, N}, 0.0, {A, A, A}, 2, StopReason::EmptyFrontier, "from the source");
    checkSpread(chain, {N, N, A}, 0.0, {A, A, A}, 2, StopReason::EmptyFrontier, "from the target");
}

// Node 0's neighbours are 1 and 2, however often and in which direction the entries list them.
void repeatedEntriesAndSelfLoops() {
    const auto graph = makeGraph(3, {{0, 0}, {0, 1}, {0, 1}, {1, 0}, {0, 2}});
    checkSpread(graph, {N, A, N}, 0.75, {N, A, N}, 0, StopReason::NoAcquisitions, "1 of 2 fails theta 0.75");
    checkSpread(graph, {N, A, N}, 0.5, {A, A, A}, 2, StopReason::EmptyFrontier, "1 of 2 meets theta 0.5");
}

// All changes of a round happen at once: in round 1 node 1 joins A and node 2 joins B. Had node 2
// seen node 1's change in the same round, it would have stayed neutral on a tie.
void synchronousUpdates() {
    const auto path = makeGraph(4, {{0, 1}, {1, 2}, {2, 3}});
    checkSpread(path, {A, N, N, B}, 0.0, {A, A, B, B}, 1, StopReason::EmptyFrontier, "two changes in one round");
}

void ownedNodesNeverChange() {
    const auto star = makeGraph(4, {{0, 1}, {0, 2}, {0, 3}});
    checkSpread(star, {A, B, B, N}, 0.0, {A, B, B, A}, 1, StopReason::EmptyFrontier, "seed facing a B majority");

    // Node 1 joins A in round 1, when nodes 2 and 3 join B; it then has a B majority but stays A.
    const auto graph = makeGraph(6, {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 5}});
    checkSpread(graph, {A, N, N, N, B, B}, 0.0, {A, A, B, B, B, B}, 1, StopReason::EmptyFrontier,
                "acquired node facing a B majority");
}

// The spread as the rule states it: every round checks every neutral node. It stopped with
// no_acquisitions if a neutral node neighbours a node that changed in the last round (or a seed,
// if none changed), since such a node was checked.
SpreadResult plainSpread(const Graph& graph, std::vector<Owner>& ownership, double theta) {
    const uint32_t n = graph.nodeCount();
    std::vector<uint32_t> lastChanged;
    for (uint32_t node = 0; node < n; ++node) {
        if (ownership[node] != N) lastChanged.push_back(node);
    }
    uint32_t rounds = 0;
    while (true) {
        std::vector<std::pair<uint32_t, Owner>> changes;
        for (uint32_t node = 0; node < n; ++node) {
            if (ownership[node] != N) continue;
            uint32_t countA = 0;
            uint32_t countB = 0;
            for (uint32_t neighbor : graph.neighbors(node)) {
                countA += ownership[neighbor] == A;
                countB += ownership[neighbor] == B;
            }
            const double needed = theta * static_cast<double>(graph.degree(node));
            if (countA > countB && static_cast<double>(countA) >= needed) changes.emplace_back(node, A);
            else if (countB > countA && static_cast<double>(countB) >= needed) changes.emplace_back(node, B);
        }
        if (changes.empty()) break;
        lastChanged.clear();
        for (const auto& [node, owner] : changes) {
            ownership[node] = owner;
            lastChanged.push_back(node);
        }
        ++rounds;
    }
    for (uint32_t node : lastChanged) {
        for (uint32_t neighbor : graph.neighbors(node)) {
            if (ownership[neighbor] == N) return {rounds, StopReason::NoAcquisitions};
        }
    }
    return {rounds, StopReason::EmptyFrontier};
}

void comparePlain(const Graph& graph, std::vector<Owner> ownership, double theta, const std::string& label) {
    auto plain = ownership;
    const auto expected = plainSpread(graph, plain, theta);
    const auto result = engine::spread(graph, ownership, theta);
    check(result.rounds == expected.rounds, label + ": rounds differ from the plain spread");
    check(result.stopReason == expected.stopReason, label + ": stop reason differs from the plain spread");
    checkEqual(ownership, plain, label + ": ownership differs from the plain spread");
}

void plainSpreadOnRandomGraphs() {
    std::mt19937 random(2718);
    for (uint32_t n = 5; n < 45; ++n) {
        Entries entries;
        for (uint32_t entry = 0; entry < 2 * n; ++entry) entries.emplace_back(random() % n, random() % n);
        const auto graph = makeGraph(n, entries);
        std::vector<Owner> ownership(n, N);
        for (uint32_t seed = 0; seed < 4; ++seed) ownership[random() % n] = seed % 2 == 0 ? A : B;
        for (double theta : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            comparePlain(graph, ownership, theta,
                         "random graph of " + std::to_string(n) + " nodes, theta " + std::to_string(theta));
        }
    }
}

// Five A seeds and then five B seeds at consecutive IDs from `offset`.
std::vector<Owner> consecutiveSeeds(uint32_t nodeCount, uint32_t offset) {
    std::vector<Owner> ownership(nodeCount, N);
    for (uint32_t i = 0; i < 5; ++i) {
        ownership[(offset + i) % nodeCount] = A;
        ownership[(offset + 5 + i) % nodeCount] = B;
    }
    return ownership;
}

void plainSpreadOnRealGraphs() {
    for (const std::string file : {"data/Erdos02.mtx", "data/ca-GrQc.mtx", "data/wiki-Vote.mtx"}) {
        const auto graph = engine::loadGraph(file);
        const uint32_t n = graph.nodeCount();
        for (double theta : {0.0, 0.5, 1.0}) {
            for (uint32_t offset : {0U, n / 3, n / 2}) {
                comparePlain(graph, consecutiveSeeds(n, offset), theta,
                             file + ", theta " + std::to_string(theta) + ", offset " + std::to_string(offset));
            }
        }
    }
}

}  // namespace

int main() {
    return runTests({
        {"threshold rule", thresholdRule},
        {"rounds and stop reasons", roundsAndStopReasons},
        {"directed chain", directedChain},
        {"repeated entries and self-loops", repeatedEntriesAndSelfLoops},
        {"synchronous updates", synchronousUpdates},
        {"owned nodes never change", ownedNodesNeverChange},
        {"plain spread on random graphs", plainSpreadOnRandomGraphs},
        {"plain spread on real graphs", plainSpreadOnRealGraphs},
    });
}
