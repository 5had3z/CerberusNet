#include "utils.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <filesystem>

bool fileExists(const std::string_view fileName, bool verbose)
{
    if (!std::filesystem::exists(std::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

std::vector<std::string> loadListFromTextFile(const std::string filename)
{
    assert(fileExists(filename, true));
    std::vector<std::string> list;
    std::ifstream f(filename);
    if (!f)
    {
        std::cout << "failed to open " << filename;
    }

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        else
            list.push_back(line);
    }

    return list;
}