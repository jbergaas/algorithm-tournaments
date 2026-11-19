#pragma once

// Plain single-thread Brandes Betweenness and ranking. The tests compare the engine's parallel
// Betweenness (algorithms.hpp) with it.

#include "engine/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <queue>
#include <stack>
#include <vector>

namespace reference {

// Unnormalised scores on the undirected view, each unordered pair counted from both ends.
inline std::vector<double> betweenness(const engine::Graph& graph) {
    const uint32_t n = graph.nodeCount();
    std::vector<double> bc(n, 0.0);
    for (uint32_t s = 0; s < n; ++s) {
        std::stack<uint32_t> S;
        std::vector<std::vector<uint32_t>> pred(n);
        std::vector<double> sigma(n, 0.0);
        std::vector<int32_t> dist(n, -1);
        std::vector<double> delta(n, 0.0);
        sigma[s] = 1.0;
        dist[s] = 0;
        std::queue<uint32_t> Q;
        Q.push(s);
        while (!Q.empty()) {
            const uint32_t v = Q.front();
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
            const uint32_t w = S.top();
            S.pop();
            for (uint32_t v : pred[w]) delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            if (w != s) bc[w] += delta[w];
        }
    }
    return bc;
}

// Nodes by decreasing score; equal scores keep increasing ID order.
inline std::vector<uint32_t> ranking(const std::vector<double>& scores) {
    std::vector<uint32_t> nodes(scores.size());
    std::iota(nodes.begin(), nodes.end(), 0);
    std::stable_sort(nodes.begin(), nodes.end(), [&](uint32_t a, uint32_t b) { return scores[a] > scores[b]; });
    return nodes;
}

}  // namespace reference
