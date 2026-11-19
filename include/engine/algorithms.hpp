#pragma once

// The five seed-selection algorithms. Each lists `count` candidate nodes, best first. All of them
// work on the undirected view of the graph. Ties are broken by the lower node ID; Coverage Greedy
// first prefers the higher degree.

#include "graph.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stack>
#include <string>
#include <vector>

namespace engine {

inline const std::array<std::string, 5> kAlgorithms = {"random", "degree", "coverage_greedy", "pagerank",
                                                       "betweenness"};

// Nodes by decreasing score; equal scores by increasing ID.
inline std::vector<uint32_t> rankByScore(const std::vector<double>& scores) {
    std::vector<uint32_t> nodes(scores.size());
    std::iota(nodes.begin(), nodes.end(), 0);
    std::sort(nodes.begin(), nodes.end(), [&](uint32_t a, uint32_t b) {
        if (scores[a] == scores[b]) return a < b;
        return scores[a] > scores[b];
    });
    return nodes;
}

// Random: a uniform shuffle of all nodes by a 64-bit Mersenne Twister seeded with `seed`.
inline std::vector<uint32_t> randomCandidates(const Graph& graph, uint32_t count, uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::vector<uint32_t> nodes(graph.nodeCount());
    std::iota(nodes.begin(), nodes.end(), 0);
    std::shuffle(nodes.begin(), nodes.end(), generator);
    if (nodes.size() > count) nodes.resize(count);
    return nodes;
}

// Degree: highest degree first.
inline std::vector<uint32_t> degreeCandidates(const Graph& graph, uint32_t count) {
    std::vector<uint32_t> nodes(graph.nodeCount());
    std::iota(nodes.begin(), nodes.end(), 0);
    auto higher = [&](uint32_t a, uint32_t b) {
        if (graph.degree(a) == graph.degree(b)) return a < b;
        return graph.degree(a) > graph.degree(b);
    };
    if (count >= nodes.size()) {
        std::sort(nodes.begin(), nodes.end(), higher);
    } else {
        std::partial_sort(nodes.begin(), nodes.begin() + count, nodes.end(), higher);
        nodes.resize(count);
    }
    return nodes;
}

// Coverage Greedy: repeatedly the node whose neighbours include the most nodes not yet
// neighbouring an earlier pick; ties by higher degree, then lower ID.
inline std::vector<uint32_t> coverageGreedyCandidates(const Graph& graph, uint32_t count) {
    const uint32_t n = graph.nodeCount();
    const uint32_t limit = std::min(count, n);
    std::vector<uint32_t> picks;
    std::vector<uint8_t> covered(n, 0);
    std::vector<uint8_t> selected(n, 0);
    while (picks.size() < limit) {
        uint32_t bestNode = n;
        uint32_t bestGain = 0;
        uint32_t bestDegree = 0;
        for (uint32_t node = 0; node < n; ++node) {
            if (selected[node]) continue;
            uint32_t gain = 0;
            for (uint32_t neighbor : graph.neighbors(node)) {
                if (!covered[neighbor]) ++gain;
            }
            const uint32_t degree = graph.degree(node);
            if (bestNode == n || gain > bestGain || (gain == bestGain && degree > bestDegree)
                || (gain == bestGain && degree == bestDegree && node < bestNode)) {
                bestNode = node;
                bestGain = gain;
                bestDegree = degree;
            }
        }
        selected[bestNode] = 1;
        picks.push_back(bestNode);
        for (uint32_t neighbor : graph.neighbors(bestNode)) covered[neighbor] = 1;
    }
    return picks;
}

// PageRank with damping 0.85, at most `iterations` power iterations (the study uses 100), and
// stopping once the scores change by at most 1e-9 in total. The rank of a node without
// neighbours is spread evenly over all nodes.
inline std::vector<double> pageRankScores(const Graph& graph, uint32_t iterations = 100) {
    constexpr double damping = 0.85;
    constexpr double tolerance = 1e-9;
    const uint32_t n = graph.nodeCount();
    if (n == 0) return {};
    const double teleport = (1.0 - damping) / static_cast<double>(n);
    std::vector<double> rank(n, 1.0 / static_cast<double>(n));
    std::vector<double> next(n, 0.0);
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        std::fill(next.begin(), next.end(), teleport);
        double danglingMass = 0.0;
        for (uint32_t node = 0; node < n; ++node) {
            const uint32_t degree = graph.degree(node);
            if (degree == 0) {
                danglingMass += rank[node];
                continue;
            }
            const double contribution = damping * rank[node] / static_cast<double>(degree);
            for (uint32_t neighbor : graph.neighbors(node)) next[neighbor] += contribution;
        }
        const double danglingContribution = damping * danglingMass / static_cast<double>(n);
        double delta = 0.0;
        for (uint32_t node = 0; node < n; ++node) {
            next[node] += danglingContribution;
            delta += std::abs(next[node] - rank[node]);
        }
        rank.swap(next);
        if (delta <= tolerance) break;
    }
    return rank;
}

inline std::vector<uint32_t> pageRankCandidates(const Graph& graph, uint32_t count) {
    auto nodes = rankByScore(pageRankScores(graph));
    if (nodes.size() > count) nodes.resize(count);
    return nodes;
}

// Exact Betweenness: Brandes' algorithm from every source, unnormalised (each unordered pair
// counted from both ends). The sources run in parallel on `threads` OpenMP threads, but their
// contributions are added in ascending source order, so the scores are bitwise the same for
// every thread count. Prints the number of finished sources to stderr about once a minute.
// The study computes the scores once per network (betweenness_ranking.hpp).
inline std::vector<double> betweennessScores(const Graph& graph, int threads) {
    using Clock = std::chrono::steady_clock;
    const uint32_t n = graph.nodeCount();
    std::vector<double> bc(n, 0.0);
    uint32_t done = 0;
    const auto start = Clock::now();
    auto lastReport = start;

    #pragma omp parallel for ordered schedule(static, 1) num_threads(threads)
    for (uint32_t s = 0; s < n; ++s) {
        std::stack<uint32_t> S;
        std::vector<std::vector<uint32_t>> pred(n);
        std::vector<double> sigma(n, 0.0);
        std::vector<int32_t> dist(n, -1);
        std::vector<double> delta(n, 0.0);
        std::vector<uint32_t> popped;
        popped.reserve(n);
        sigma[s] = 1.0;
        dist[s] = 0;
        std::queue<uint32_t> Q;
        Q.push(s);
        while (!Q.empty()) {
            uint32_t v = Q.front();
            Q.pop();
            S.push(v);
            for (uint32_t w : graph.neighbors(v)) {
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    Q.push(w);
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }
        while (!S.empty()) {
            uint32_t w = S.top();
            S.pop();
            for (uint32_t v : pred[w]) {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            popped.push_back(w);
        }

        #pragma omp ordered
        {
            for (uint32_t w : popped) {
                if (w != s) bc[w] += delta[w];
            }
            ++done;
            if (Clock::now() - lastReport >= std::chrono::minutes(1)) {
                lastReport = Clock::now();
                const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(lastReport - start).count();
                std::cerr << "Betweenness: " << done << " of " << n << " sources done after " << seconds << " s"
                          << std::endl;
            }
        }
    }
    return bc;
}

}  // namespace engine
