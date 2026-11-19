// Random's match seeds (seeds.hpp): the FNV-1a hash, pinned seeds, and that each input of the
// seed changes it. That the opponent, the sequence, the threshold and the file's location do not
// change it is checked on the seeds runBatch records (test_batch.cpp).

#include "check.hpp"
#include "engine/dataset_identity.hpp"
#include "engine/seeds.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

uint64_t fnv1a(std::string_view text) {
    engine::Fnv1a hash(0xcbf29ce484222325ULL);
    for (unsigned char byte : text) hash.addByte(byte);
    return hash.value();
}

// The published FNV-1a 64-bit test values, from the standard offset basis.
void hash() {
    check(fnv1a("") == 0xcbf29ce484222325ULL, "FNV-1a of the empty text");
    check(fnv1a("a") == 0xaf63dc4c8601ec8cULL, "FNV-1a of \"a\"");
    check(fnv1a("foobar") == 0x85944171f73967e8ULL, "FNV-1a of \"foobar\"");
}

// The dataset identity of the ten-node test graph.
std::string testIdentity() {
    const std::string identity = engine::computeDatasetIdentity("data/test.mtx", 10);
    check(identity == "ci-dataset-v1:3014655995f347de3c4ae3b231cf82b9f3c8aca4d7d453e151d1f84c28dd3edc:"
                      "10b4796eac59c7d81c33711f219ba227247a4e338adad078159ba01e87590841:10",
          "identity of data/test.mtx");
    return identity;
}

void pinnedSeeds() {
    const auto identity = testIdentity();
    check(engine::matchSeed(42, identity, "random", 3, 0, 'A') == 14454494734286381522ULL, "base seed 42, player A");
    check(engine::matchSeed(42, identity, "random", 3, 0, 'B') == 14454493634774753311ULL, "base seed 42, player B");
    check(engine::matchSeed(20260810, identity, "random", 3, 7, 'A') == 12974688659692391042ULL,
          "base seed 20260810, run 7, player A");
}

void inputsChangeTheSeed() {
    const auto identity = testIdentity();
    const uint64_t seed = engine::matchSeed(20260810, identity, "random", 3, 7, 'A');
    const std::vector<std::pair<std::string, uint64_t>> changed = {
        {"base seed", engine::matchSeed(20260811, identity, "random", 3, 7, 'A')},
        {"high bits of the base seed", engine::matchSeed(20260810 + (1ULL << 40), identity, "random", 3, 7, 'A')},
        {"identity", engine::matchSeed(20260810, identity + "x", "random", 3, 7, 'A')},
        {"own algorithm", engine::matchSeed(20260810, identity, "degree", 3, 7, 'A')},
        {"budget", engine::matchSeed(20260810, identity, "random", 4, 7, 'A')},
        {"run", engine::matchSeed(20260810, identity, "random", 3, 8, 'A')},
        {"player", engine::matchSeed(20260810, identity, "random", 3, 7, 'B')},
    };
    std::set<uint64_t> distinct = {seed};
    for (const auto& [input, value] : changed) {
        check(value != seed, "the " + input + " must change the seed");
        distinct.insert(value);
    }
    check(distinct.size() == changed.size() + 1, "every changed input gives its own seed");
}

}  // namespace

int main() {
    return runTests({
        {"hash", hash},
        {"pinned seeds", pinnedSeeds},
        {"inputs change the seed", inputsChangeTheSeed},
    });
}
