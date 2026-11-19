#pragma once

// Random's seed in a match: a hash of the base seed, the graph's dataset identity
// (dataset_identity.hpp), the player's own algorithm, the budget, the run ID and the player
// ('A' or 'B'). The opponent, the picking sequence and the threshold do not enter it, so in a
// given run Random draws the same candidates against every opponent, under every sequence and at
// every threshold. Every algorithm gets a seed, but only Random uses it.

#include <cstdint>
#include <string_view>

namespace engine {

// FNV-1a, 64 bit. Integers are fed one byte at a time from the lowest byte up, strings with
// their length first, so the value is the same on every machine and compiler.
class Fnv1a {
    static constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t value_;

public:
    explicit Fnv1a(uint64_t start) : value_(start) {}

    void addByte(uint8_t byte) {
        value_ ^= byte;
        value_ *= kPrime;
    }

    void addInteger(uint64_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) addByte(static_cast<uint8_t>(value >> (8 * i)));
    }

    void addString(std::string_view text) {
        addInteger(text.size(), 8);
        for (unsigned char byte : text) addByte(byte);
    }

    uint64_t value() const { return value_; }
};

// The state the seed hash starts from. Changing it changes every seed.
inline constexpr uint64_t kSeedHashStart = 0xaafb30a1ab5006feULL;

inline uint64_t matchSeed(uint64_t baseSeed, std::string_view datasetIdentity, std::string_view algorithm,
                          uint32_t budget, uint32_t run, char player) {
    Fnv1a hash(kSeedHashStart);
    hash.addInteger(baseSeed, 8);
    hash.addString(datasetIdentity);
    hash.addString(algorithm);
    hash.addInteger(budget, 4);
    hash.addInteger(run, 4);
    hash.addByte(static_cast<uint8_t>(player));
    return hash.value();
}

}  // namespace engine
