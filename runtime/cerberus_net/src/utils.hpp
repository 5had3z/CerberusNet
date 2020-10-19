#pragma once

#include <string>
#include <vector>

std::vector<std::string> loadListFromTextFile(const std::string filename);
bool fileExists(const std::string_view fileName, bool verbose);
