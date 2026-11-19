#pragma once

// Helpers shared by the test programs. A test is a function that makes checks; a failed check
// throws, and runTests() reports it and goes on with the next test.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

inline unsigned checkCount = 0;

// Counts one check and throws `message` if it fails.
inline void check(bool condition, const std::string& message) {
    ++checkCount;
    if (!condition) throw std::runtime_error(message);
}

template <class T>
inline constexpr bool kIsVector = false;

template <class T>
inline constexpr bool kIsVector<std::vector<T>> = true;

// The values as "[1,2,3]"; enums as their numbers, nested vectors as nested lists.
template <class T>
std::string str(const std::vector<T>& values) {
    std::ostringstream text;
    text.precision(17);
    text << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) text << ',';
        if constexpr (std::is_enum_v<T>) text << static_cast<int>(values[i]);
        else if constexpr (kIsVector<T>) text << str(values[i]);
        else text << values[i];
    }
    text << ']';
    return text.str();
}

// Checks that two vectors are equal. A failure shows both when they are short, otherwise the
// first index where they differ.
template <class T>
void checkEqual(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& label) {
    ++checkCount;
    if (actual == expected) return;
    if (actual.size() + expected.size() <= 64) {
        throw std::runtime_error(label + ": expected " + str(expected) + ", got " + str(actual));
    }
    const auto index = std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end()).first
                     - actual.begin();
    throw std::runtime_error(label + ": differs at index " + std::to_string(index) + " (sizes "
                             + std::to_string(actual.size()) + " and " + std::to_string(expected.size()) + ")");
}

// Checks that `function` throws an exception whose message contains `expected`.
template <class Function>
void rejects(Function function, const std::string& label, const std::string& expected = "") {
    ++checkCount;
    try {
        function();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(expected) == std::string::npos) {
            throw std::runtime_error(label + ": rejected with \"" + error.what() + "\", expected \"" + expected + "\"");
        }
        return;
    }
    throw std::runtime_error(label + ": accepted");
}

// Runs the tests in order, prints one line for each and returns the exit code for main().
inline int runTests(std::initializer_list<std::pair<const char*, std::function<void()>>> tests) {
    int failures = 0;
    for (const auto& [name, test] : tests) {
        const unsigned before = checkCount;
        try {
            test();
            std::cout << "PASS " << name << " (" << checkCount - before << " checks)\n";
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

// A new empty directory in the temporary directory ($TMPDIR), removed with the object.
class ScratchDirectory {
    std::filesystem::path path_;

public:
    ScratchDirectory() {
        std::string pattern = (std::filesystem::temp_directory_path() / "engine-test-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("cannot create a scratch directory");
        path_ = pattern;
    }

    ~ScratchDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;

    std::filesystem::path operator/(const std::string& name) const { return path_ / name; }
    const std::filesystem::path& path() const { return path_; }
};

// The values as 32-bit little-endian integers, the layout of a stored ranking.
inline std::string littleEndian(const std::vector<uint32_t>& values) {
    std::string bytes;
    for (const uint32_t value : values) {
        for (unsigned byte = 0; byte < 4; ++byte) bytes.push_back(static_cast<char>(value >> (8 * byte)));
    }
    return bytes;
}

inline std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

inline std::filesystem::path writeFile(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot write " + path.string());
    }
    return path;
}
