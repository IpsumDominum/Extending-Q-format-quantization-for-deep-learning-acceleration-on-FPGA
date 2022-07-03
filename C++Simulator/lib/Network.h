#pragma once
#include <cstdint>
#include "qLib8.h"
#include <iostream>
#include "fixed8.h"
#include <vector>
#include "json.h"
#include <fstream>
#include <string>
#include "Layers/Layers.h"

using json = nlohmann::json;

class Network
{
    private:        
        std::vector<Layer*> layers;
        bool debug_or_not = false;
    public:
        //Constructor        
        Network();
        void build_from_json(std::string json_path);
        int inference(std::vector<std::vector<fixed8>> input_matrix,std::vector<std::vector<long double>> input_matrix_double);
        int predict_D(std::vector<long double> network_output);
        int predict(std::vector<fixed8> network_output);
        void addLayer(LayerType layer_type,json layerInfo);
        
        //Cast fixed -> double
        std::vector<std::vector<std::vector<long double>>> cast_fixed_to_double_3D(std::vector<std::vector<std::vector<fixed8>>> to_be_cast);
        std::vector<long double> cast_fixed_to_double_1D(std::vector<fixed8> to_be_cast);
        //Cast double -> fixed
        std::vector<std::vector<std::vector<fixed8>>> cast_double_to_fixed_3D(std::vector<std::vector<std::vector<long double>>> to_be_cast,int cast_m, int cast_n);
        std::vector<fixed8> cast_double_to_fixed_1D(std::vector<long double> to_be_cast, int cast_m, int cast_n);
        void debug(bool debug_set);
};
