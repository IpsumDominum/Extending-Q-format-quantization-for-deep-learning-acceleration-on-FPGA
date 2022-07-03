#pragma once
#include <cstdint>
#include "../qLib8.h"
#include <iostream>
#include "../fixed8.h"
#include <vector>
#include "../json.h"
#include <fstream>
#include <string>
#include <algorithm>    // std::max

using json = nlohmann::json;
 
enum LayerType { CONV = 0, DENSE = 1, BATCHNORM_3D = 2,BATCHNORM_1D, RELU=3 ,MAXPOOL=4 ,FLAT=5 }; 

// Base class
class Layer {
    public:
        LayerType layer_type;
        std::string layer_name = "unknown";
        bool use_fixed = true;
        bool debug_or_not = false;
        bool write_output = false;
        int out_m; 
        int out_n;     
        std::string output_dir = "outputs";
        virtual std::vector<std::vector<std::vector<fixed8>>> inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix){            
            std::vector<std::vector<std::vector<fixed8>>> res;
            return res;
        };
        virtual std::vector<fixed8> inference3D21D(std::vector<std::vector<std::vector<fixed8>>> input_matrix){
            std::vector<fixed8> res;
            return res;
        };
        virtual std::vector<fixed8> inference1D21D(std::vector<fixed8> input_matrix){
            std::vector<fixed8> res;
            return res;
        };
        virtual std::vector<std::vector<std::vector<long double>>> inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix){
            std::vector<std::vector<std::vector<long double>>> res;
            return res;
        }
        virtual std::vector<long double> inference3D21D_double(std::vector<std::vector<std::vector<long double>>> input_matrix){
            std::vector<long double> res;
            return res;
        };
        virtual std::vector<long double> inference1D21D_double(std::vector<long double> input_matrix){
            std::vector<long double> res;
            return res;
        };
        void write_output_3D(std::vector<std::vector<std::vector<fixed8>>> output);
        void write_output_1D(std::vector<fixed8> output);
        //Write double
        void write_output_3D_double(std::vector<std::vector<std::vector<long double>>> output);
        void write_output_1D_double(std::vector<long double> output);
};

// Convolutional layer
class ConvLayer: public Layer {
    private:
        int conv_dim_in;
        int conv_dim_out; 
        int conv_strides_x;
        int conv_strides_y;
        int conv_kernel_size_x;
        int conv_kernel_size_y;
        int expected_input_dim;
        int output_dim;        
        int weight_m;
        int weight_n;
        int bias_m;
        int bias_n;
        bool truncate_or_not;
        int truncate_m;
        int truncate_n;   
        std::string activation;
        std::vector<std::vector<std::vector<std::vector<fixed8>>>> weight {};
        std::vector<std::vector<std::vector<std::vector<long double>>>> weight_double {};
        std::vector<fixed8> bias{};
        std::vector<long double> bias_double{};
   public:
        virtual std::vector<std::vector<std::vector<fixed8>>> inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix);
        virtual std::vector<std::vector<std::vector<long double>>> inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix);
        //Constructor
        ConvLayer(json layerInfo,bool debug_set);
};

// Dense Layer
class DenseLayer: public Layer {
    private:        
        int input_nodes_dim;
        int output_nodes_dim;
        int weight_m;
        int weight_n;
        int bias_m;
        int bias_n;            
        bool truncate_or_not;
        int truncate_m;
        int truncate_n;
        std::string activation;
        std::vector<std::vector<fixed8>> weight {};
        std::vector<std::vector<long double>> weight_double {};
        std::vector<fixed8> bias {};
        std::vector<long double> bias_double {};
   public:        
        virtual std::vector<fixed8> inference1D21D(std::vector<fixed8> input_matrix);
        virtual std::vector<long double> inference1D21D_double(std::vector<long double> input_matrix);
        //Constructor
        DenseLayer(json layerInfo,bool debug_set);
};

// Batch Norm Layer
class BatchNormLayer: public Layer {
    private:
        int inv_std_m;
        int inv_std_n;
        int neg_mean_m;
        int neg_mean_n;
        int batch_norm_dim;
        bool truncate_or_not;
        int truncate_m;
        int truncate_n;   
        std::string activation;
        std::vector<fixed8> batch_inv_std {};
        std::vector<fixed8> batch_neg_mean{};
        std::vector<long double> batch_inv_std_double {};
        std::vector<long double> batch_neg_mean_double {};
   public:        
        //Fixed inference
        virtual std::vector<std::vector<std::vector<fixed8>>> inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix);
        virtual std::vector<fixed8> inference1D21D(std::vector<fixed8> input_matrix);
        //Double inference
        virtual std::vector<std::vector<std::vector<long double>>> inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix);
        virtual std::vector<long double> inference1D21D_double(std::vector<long double> input_matrix);
        //Constructor
        BatchNormLayer(json layerInfo,std::string batch_norm_specification,bool debug_set);
};

// Batch Norm Layer
class MaxPoolLayer: public Layer {
    private:
        bool truncate_or_not;
        int truncate_m;
        int truncate_n;   
        std::string activation;
   public:        
        virtual std::vector<std::vector<std::vector<fixed8>>> inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix);
        virtual std::vector<std::vector<std::vector<long double>>> inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix);
        //Constructor
        MaxPoolLayer(json layerInfo,bool debug_set);
        double maxD(double a,double b, double c,double d);
};

// Flatten Layer
class FlatLayer: public Layer {
    private:        
        bool truncate_or_not;
        int truncate_m;
        int truncate_n;   
        std::string activation;        
   public:        
        virtual std::vector<fixed8> inference3D21D(std::vector<std::vector<std::vector<fixed8>>> input_matrix);
        virtual std::vector<long double> inference3D21D_double(std::vector<std::vector<std::vector<long double>>> input_matrix);
        //Constructor
        FlatLayer(json layerInfo,bool debug_set);
};