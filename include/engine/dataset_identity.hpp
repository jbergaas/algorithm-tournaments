#pragma once

// The dataset identity of a graph file enters Random's seeds (seeds.hpp), so Random
// draws the same candidates for the same graph wherever the file lies. It is the text
//   ci-dataset-v1:<SHA-256 of the file>:<SHA-256 of the node IDs>:<n>
// where the node IDs 0..n-1 are hashed as 32-bit little-endian integers. The prefix and the
// layout are fixed inputs of the seed hash; changing them changes every seed.

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

// SHA-256 (FIPS 180-4).
class Sha256 {
    static constexpr std::array<uint32_t, 64> kRoundConstants = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t length_ = 0;

    static uint32_t rotate(uint32_t value, int bits) { return (value >> bits) | (value << (32 - bits)); }

    void compress() {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            w[i] = uint32_t(block_[4 * i]) << 24 | uint32_t(block_[4 * i + 1]) << 16
                 | uint32_t(block_[4 * i + 2]) << 8 | uint32_t(block_[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g))
                              + kRoundConstants[i] + w[i];
            const uint32_t t2 = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        const std::array<uint32_t, 8> add = {a, b, c, d, e, f, g, h};
        for (int i = 0; i < 8; ++i) state_[i] += add[i];
    }

public:
    void update(const uint8_t* data, size_t size) {
        length_ += size;
        for (size_t i = 0; i < size; ++i) {
            block_[used_++] = data[i];
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }

    // The digest as 64 lower-case hex digits; the object cannot be updated afterwards.
    std::string hexDigest() {
        const uint64_t bits = length_ * 8;
        const uint8_t one = 0x80;
        const uint8_t zero = 0;
        update(&one, 1);
        while (used_ != 56) update(&zero, 1);
        for (int shift = 56; shift >= 0; shift -= 8) {
            const auto byte = static_cast<uint8_t>(bits >> shift);
            update(&byte, 1);
        }
        std::string hex;
        char digits[9];
        for (const uint32_t word : state_) {
            std::snprintf(digits, sizeof digits, "%08x", word);
            hex += digits;
        }
        return hex;
    }
};

inline std::string sha256File(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!std::filesystem::is_regular_file(path) || !input) throw std::runtime_error("Cannot open file: " + path);
    Sha256 hash;
    std::vector<char> buffer(1 << 16);
    while (input.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || input.gcount() > 0) {
        hash.update(reinterpret_cast<const uint8_t*>(buffer.data()), static_cast<size_t>(input.gcount()));
    }
    return hash.hexDigest();
}

// SHA-256 of the node IDs 0..n-1 as 32-bit little-endian integers.
inline std::string sha256NodeIds(uint32_t nodeCount) {
    Sha256 hash;
    for (uint32_t id = 0; id < nodeCount; ++id) {
        const uint8_t bytes[4] = {uint8_t(id), uint8_t(id >> 8), uint8_t(id >> 16), uint8_t(id >> 24)};
        hash.update(bytes, 4);
    }
    return hash.hexDigest();
}

inline std::string computeDatasetIdentity(const std::string& graphPath, uint32_t nodeCount) {
    return "ci-dataset-v1:" + sha256File(graphPath) + ":" + sha256NodeIds(nodeCount) + ":"
         + std::to_string(nodeCount);
}

}  // namespace engine
