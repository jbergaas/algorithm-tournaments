// Dataset identities (dataset_identity.hpp): SHA-256 against published test values, the hash
// of the node IDs, and the identity of a small file, which depends on the bytes but not on the
// file's name or folder.

#include "check.hpp"
#include "engine/dataset_identity.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string sha256(const std::string& text) {
    engine::Sha256 hash;
    hash.update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    return hash.hexDigest();
}

void sha256Values() {
    check(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 of \"abc\"");
    check(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
              == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "SHA-256 of the two-block message");

    ScratchDirectory scratch;
    check(engine::sha256File(writeFile(scratch / "abc with space and\nnewline", "abc").string())
              == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 of a file holding \"abc\"");
    check(engine::sha256File(writeFile(scratch / "empty", "").string())
              == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "SHA-256 of an empty file");
    check(engine::sha256File(writeFile(scratch / "million", std::string(1000000, 'a')).string())
              == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "SHA-256 of a million 'a', longer than the read buffer");
    rejects([&] { engine::sha256File((scratch / "missing").string()); }, "missing file", "Cannot open");
    rejects([&] { engine::sha256File(scratch.path().string()); }, "folder", "Cannot open");
}

void nodeIds() {
    const std::vector<std::pair<uint32_t, std::string>> expected = {
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {1, "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"},
        {2, "01acecb507abfe1a354aa8064f4af5d3f1acd019e37db3c11c97523b71c76e9d"},
        {3, "ad5dc1478de06a4c2728ea528bd9361a4b945e92a414bf4d180cedaaeaa5f4cc"},
        {17, "3ef6f38adb85f46f95c0597848fda1b8e74e65c025e441c66700d9802fa6e085"},
    };
    for (const auto& [count, hash] : expected) {
        check(engine::sha256NodeIds(count) == hash, "SHA-256 of the IDs of " + std::to_string(count) + " nodes");
    }
}

void identity() {
    ScratchDirectory scratch;
    const std::string bytes = "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 1\n1 2\n";
    const std::string expected = "ci-dataset-v1:399148a0889a8767dd2beccde2adfb95275e2fd7dcbc12edd584680a52f917b5:"
                                 "ad5dc1478de06a4c2728ea528bd9361a4b945e92a414bf4d180cedaaeaa5f4cc:3";
    check(engine::computeDatasetIdentity(writeFile(scratch / "graph.mtx", bytes).string(), 3) == expected,
          "identity of the three-node file");
    std::filesystem::create_directory(scratch / "elsewhere");
    check(engine::computeDatasetIdentity(writeFile(scratch / "elsewhere" / "renamed.mtx", bytes).string(), 3)
              == expected,
          "the identity does not depend on the file's name or folder");
    check(engine::computeDatasetIdentity(writeFile(scratch / "changed.mtx", bytes + "\n").string(), 3) != expected,
          "the identity depends on the file's bytes");
    rejects([&] { engine::computeDatasetIdentity((scratch / "missing.mtx").string(), 3); }, "missing file",
            "Cannot open");
}

}  // namespace

int main() {
    return runTests({
        {"sha256", sha256Values},
        {"node IDs", nodeIds},
        {"identity", identity},
    });
}
