#pragma once
#include <cstdint>
#include <iostream>
#include "../fixed8.h"
#include <vector>
#include "../json.h"
#include "../Network.h"
#include <fstream>
#include <string>

struct MNISTDataset{
    std::vector<std::vector<std::vector<fixed8>>> ImageData;
    std::vector<std::vector<std::vector<long double>>> ImageData_double;
    std::vector<double> labels;
};
MNISTDataset load_mnist(std::string data_json_path);
