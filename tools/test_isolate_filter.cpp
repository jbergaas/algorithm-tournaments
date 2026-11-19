// The isolate filter (preprocessing/isolate_filter.hpp): it removes exactly the nodes without an
// undirected neighbour, keeps every neighbourhood (renumbered in increasing old-ID order), leaves
// the spread unchanged, and does nothing to a graph without such nodes.

#include "check.hpp"
#include "isolate_filter.hpp"
#include "engine/graph.hpp"
#include "engine/spread.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using engine::Owner;

void checkRenumbering(const engine::Graph& before, const engine::Graph& after, const isolate_filter::Result& result) {
    check(result.oldToNew.size() == before.nodeCount() && result.newToOld.size() == after.nodeCount(), "mapping sizes");
    uint32_t next = 0;
    for (uint32_t old = 0; old < before.nodeCount(); ++old) {
        if (before.degree(old) == 0) {
            check(result.oldToNew[old] == isolate_filter::removedId, "node " + std::to_string(old) + " is kept");
            continue;
        }
        const auto id = result.oldToNew[old];
        check(id == next++ && result.newToOld[id] == old, "the kept nodes keep their order");
        std::vector<uint32_t> expected;
        for (uint32_t neighbor : before.neighbors(old)) expected.push_back(result.oldToNew[neighbor]);
        const auto neighbors = after.neighbors(id);
        checkEqual(std::vector<uint32_t>(neighbors.begin(), neighbors.end()), expected,
                   "neighbours of node " + std::to_string(old));
    }
}

void checkSpread(const engine::Graph& before, const engine::Graph& after, const isolate_filter::Result& result) {
    for (double theta : {0.0, 0.5, 1.0}) {
        std::vector<Owner> original(before.nodeCount(), Owner::Neutral);
        std::vector<Owner> filtered(after.nodeCount(), Owner::Neutral);
        for (uint32_t old : {0U, 2U}) {
            original[old] = Owner::A;
            filtered[result.oldToNew[old]] = Owner::A;
        }
        original[6] = Owner::B;
        filtered[result.oldToNew[6]] = Owner::B;
        const auto a = engine::spread(before, original, theta);
        const auto b = engine::spread(after, filtered, theta);
        for (uint32_t id = 0; id < after.nodeCount(); ++id) {
            check(filtered[id] == original[result.newToOld[id]], "ownership differs after filtering");
        }
        check(a.rounds == b.rounds && a.stopReason == b.stopReason, "the spread differs after filtering");
    }
}

void removesIsolatedNodes() {
    ScratchDirectory scratch;
    // Repeated, reciprocal and loop entries; node 4 (0-based) has only an entry from node 3, and
    // node 7 has only loops.
    const auto input = writeFile(scratch / "mixed.mtx", "%%MatrixMarket matrix coordinate pattern general\n"
                                                        "12 12 9\n1 2\n1 2\n2 1\n2 2\n3 4\n4 5\n6 7\n8 8\n8 8\n");
    const auto bytes = readFile(input);
    const auto original = engine::loadGraph(input.string());
    check(original.degree(4) == 1, "a node with only an incoming entry has a neighbour");
    check(original.edgeCount() == 4, "repeated, reciprocal and loop entries collapse to 4 edges");

    const auto result = isolate_filter::run(input.string(), scratch / "mixed-output");
    check(result.changed(), "the filter removed nodes");
    checkEqual(result.removed, {7, 8, 9, 10, 11}, "exactly the nodes without a neighbour are removed");
    const auto filtered = engine::loadGraph(result.graphPath);
    check(filtered.nodeCount() == 7 && filtered.entryCount() == 8 && filtered.edgeCount() == 4,
          "the filtered file lists each of the 4 edges once, read both ways");
    checkRenumbering(original, filtered, result);
    checkSpread(original, filtered, result);
    check(readFile(input) == bytes, "the input file changed");
    check(readFile(scratch / "mixed-output" / "old_new.tsv")
              .starts_with("old_internal_id_0based\tnew_internal_id_0based\n0\t0\n"),
          "mapping file");
    check(readFile(scratch / "mixed-output" / "removed.tsv") == "old_internal_id_0based\n7\n8\n9\n10\n11\n",
          "removed file");
    check(readFile(scratch / "mixed-output" / "report.json").find("\"isolated_nodes_removed\":5,\"nodes_after\":7")
              != std::string::npos,
          "report");

    const auto second = isolate_filter::run(result.graphPath, scratch / "second-output");
    check(!second.changed() && second.graphPath == result.graphPath
              && !fs::exists(scratch / "second-output" / "filtered.mtx"),
          "a second pass changes nothing");
    rejects([&] { isolate_filter::run(input.string(), scratch / "mixed-output"); }, "existing output folder",
            "exists");
}

void refusesEmptyResults() {
    ScratchDirectory scratch;
    const auto allIsolated = writeFile(scratch / "all-isolated.mtx",
                                       "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n2 2\n");
    rejects([&] { isolate_filter::run(allIsolated.string(), scratch / "empty-output"); }, "only isolated nodes",
            "empty");
    check(!fs::exists(scratch / "empty-output"), "a refused run wrote an output folder");
    const auto noNodes = writeFile(scratch / "no-nodes.mtx",
                                   "%%MatrixMarket matrix coordinate pattern general\n0 0 0\n");
    rejects([&] { isolate_filter::run(noNodes.string(), scratch / "zero-output"); }, "no nodes", "empty");
}

}  // namespace

int main() {
    return runTests({
        {"removes isolated nodes", removesIsolatedNodes},
        {"refuses empty results", refusesEmptyResults},
    });
}
