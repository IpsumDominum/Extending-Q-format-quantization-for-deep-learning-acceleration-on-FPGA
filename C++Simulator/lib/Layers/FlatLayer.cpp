#include "Layers.h"
#include "../json.h"
#include <vector>
#include <string>

std::vector<fixed8> FlatLayer::inference3D21D(std::vector<std::vector<std::vector<fixed8>>> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Flat Layer inference"<<"\n";
        std::cout<<"Input matrix shape : ["<<input_matrix.size() <<","<<input_matrix[0].size()<<","<<input_matrix[0][0].size()<<"] \n";
    }
    std::vector<fixed8> output {};    
    //Flatten Operation Inference    
    int input_size_3 = input_matrix.size(); //50
    int input_size_2 = input_matrix[0].size(); //4
    int input_size_1 = input_matrix[0][0].size(); //4

    int cnt = 0;
    for (int i= 0; i < input_size_1; i++){
        for (int j = 0; j < input_size_2; j++){
            for (int k = 0; k < input_size_3; k++){
				output.push_back(input_matrix[k][i][j]);                                
                if(truncate_or_not==true){
                    output[cnt] = truncate(output[cnt], truncate_m, truncate_n);
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[cnt].bin() & 0b1000) != 0)
                            output[cnt].reset(truncate_m, truncate_n); //Relu
                    }                
                }else{
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[cnt].bin() & 0b1000) != 0)
                            output[cnt].reset(output[cnt].m(), output[cnt].n()); //Relu
                    }
                }
                cnt +=1;
            }
        }
    }
    //Inference Complete... Output ready to be retreived
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Output Shape: ["<<output.size()<<"]"<<"\n";
        std::cout<<"================================"<<"\n";
    }    
    if(write_output==true){
        write_output_1D(output);
    }
    return output;
}


// ===================================================
// DOUBLE INFERENCE
// ===================================================

std::vector<long double> FlatLayer::inference3D21D_double(std::vector<std::vector<std::vector<long double>>> input_matrix){
//DEBUG
    if(debug_or_not==true){
        std::cout<<"Flat Layer inference"<<"\n";
        std::cout<<"Input matrix shape : ["<<input_matrix.size() <<","<<input_matrix[0].size()<<","<<input_matrix[0][0].size()<<"] \n";
    }    
    std::vector<long double> output {};    
    //Flatten Operation Inference    
    int input_size_3 = input_matrix.size(); //50
    int input_size_2 = input_matrix[0].size(); //4
    int input_size_1 = input_matrix[0][0].size(); //4

    int cnt = 0;

    if(strcmp(activation.data(),"relu")==0){
        if(debug_or_not==true){
            std::cout<<"RELU"<<"\n";
        }
    }
    for (int i= 0; i < input_size_1; i++){
        for (int j = 0; j < input_size_2; j++){
            for (int k = 0; k < input_size_3; k++){
				output.push_back(input_matrix[k][i][j]);                
                if(strcmp(activation.data(),"relu")==0){                    
                    if (output[cnt]<0){
                        output[cnt] = 0; //RELU
                    }
                }                
                cnt +=1;
            }
        }
    }
    //Inference Complete... Output ready to be retreived
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Output Shape: ["<<output.size()<<"]"<<"\n";
        std::cout<<"================================"<<"\n";
    }    
    if(write_output==true){
        write_output_1D_double(output);
    }
    return output;
}

FlatLayer::FlatLayer(json layerInfo,bool debug_set){
    //====================================
    //DEBUG
    if(debug_set==true){
        std::cout<<"Flat Layer"<< layerInfo<<"\n";
    }
    debug_or_not = debug_set;
    //====================================
    out_m = layerInfo["out_Q_m"];
    out_n = layerInfo["out_Q_n"];
    truncate_m = layerInfo["truncate_m"];
    truncate_n = layerInfo["truncate_n"];;
    activation = layerInfo["activation"];
    use_fixed = layerInfo["use_fixed"];
    truncate_or_not = layerInfo["truncate"];
    layer_name = layerInfo["name"];
    layer_type = FLAT;
    write_output = layerInfo["write_output"];
}