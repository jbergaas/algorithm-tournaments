// isolate_filter --input GRAPH.mtx --output-dir NEW_FOLDER
// Removes the nodes without edges (see isolate_filter.hpp) and prints the report.

#include "isolate_filter.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const char* usage = "Usage: isolate_filter --input GRAPH.mtx --output-dir NEW_FOLDER\n";
    if (argc != 5 || std::string(argv[1]) != "--input" || std::string(argv[3]) != "--output-dir") {
        std::cerr << usage;
        return 1;
    }
    try {
        std::cout << isolate_filter::reportJson(isolate_filter::run(argv[2], argv[4]));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Isolate filter failed: " << error.what() << '\n';
        return 1;
    }
}
