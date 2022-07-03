#include "Layers.h"
#include "../json.h"
#include <vector>
#include <string>

std::vector<std::vector<std::vector<fixed8>>> MaxPoolLayer::inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Max Pool Layer inference"<<"\n";
        std::cout<<"Input matrix shape : ["<<input_matrix.size() <<","<<input_matrix[0].size()<<","<<input_matrix[0][0].size()<<"] \n";
    }
    std::vector<std::vector<std::vector<fixed8>>> output {};    
    //Max Pool Inference    

    int max_pool_stride_x = input_matrix[0].size()/2;
    int max_pool_stride_y = input_matrix[0][0].size()/2;

	for (std::vector<std::vector<std::vector<fixed8>>>::size_type i = 0; i < input_matrix.size(); i++){
        std::vector<std::vector<fixed8>> v {};
        output.push_back(v); 
		for (int j = 0; j < max_pool_stride_x; j++){
            std::vector<fixed8> vv {};
            output[i].push_back(vv); 
			for (int k = 0; k < max_pool_stride_y; k++)
			{
                /*Max Pool Operation*/
				output[i][j].push_back(maxQ(input_matrix[i][2 * j][2 * k], input_matrix[i][2 * j + 1][2 * k], input_matrix[i][2 * j][2 * k + 1], input_matrix[i][2 * j + 1][2 * k + 1]));            
                /*Activation*/
                if(truncate_or_not==true){
                    output[i][j][k] = truncate(output[i][j][k], truncate_m, truncate_n);
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[i][j][k].bin() & 0b1000) != 0)
                            output[i][j][k].reset(truncate_m, truncate_n); //Relu
                    }                
                }else{
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[i][j][k].bin() & 0b1000) != 0)
                            output[i][j][k].reset(output[i][j][k].m(), output[i][j][k].n()); //Relu
                    }
                }


			}
        }
    }
    //Inference Complete... Output ready to be retreived
    if(debug_or_not==true){
        std::cout<<"Output matrix shape : ["<<output.size() <<","<<output[0].size()<<","<<output[0][0].size()<<"] \n";
        std::cout<<"================================"<<"\n";
    }
    if(write_output==true){
        write_output_3D(output);
    }
    return output;
}


// ===================================================
// DOUBLE INFERENCE
// ===================================================

std::vector<std::vector<std::vector<long double>>> MaxPoolLayer::inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix){
//DEBUG
    if(debug_or_not==true){
        std::cout<<"Max Pool Layer inference (Double)"<<"\n";
        std::cout<<"Input matrix shape : ["<<input_matrix.size() <<","<<input_matrix[0].size()<<","<<input_matrix[0][0].size()<<"] \n";
    }
    std::vector<std::vector<std::vector<long double>>> output {};    
    //Max Pool Inference    

    int max_pool_stride_x = input_matrix[0].size()/2;
    int max_pool_stride_y = input_matrix[0][0].size()/2;

    if(strcmp(activation.data(),"relu")==0){
        if(debug_or_not==true){
            std::cout<<"RELU"<<"\n";
        }
    }
	for (std::vector<std::vector<std::vector<long double>>>::size_type i = 0; i < input_matrix.size(); i++){
        std::vector<std::vector<long double>> v {};
        output.push_back(v); 
		for (int j = 0; j < max_pool_stride_x; j++){
            std::vector<long double> vv {};
            output[i].push_back(vv); 
			for (int k = 0; k < max_pool_stride_y; k++)
			{
                /*Max Pool Operation*/
				output[i][j].push_back(maxD(input_matrix[i][2 * j][2 * k], input_matrix[i][2 * j + 1][2 * k], input_matrix[i][2 * j][2 * k + 1], input_matrix[i][2 * j + 1][2 * k + 1]));            
                if(strcmp(activation.data(),"relu")==0){                    
                    if (output[i][j][k]<0){
                        output[i][j][k] = 0; //RELU
                    }
                }                
			}
        }
    }
    //Inference Complete... Output ready to be retreived
    if(debug_or_not==true){
        std::cout<<"Output matrix shape : ["<<output.size() <<","<<output[0].size()<<","<<output[0][0].size()<<"] \n";
        std::cout<<"================================"<<"\n";
    }
     if(write_output==true){
        write_output_3D_double(output);
    }
    return output;
}

double MaxPoolLayer::maxD(double a,double b, double c,double d)
{
	return std::max(std::max(a,b),std::max(c,d));
}
MaxPoolLayer::MaxPoolLayer(json layerInfo,bool debug_set){
    debug_or_not = debug_set;
    //====================================
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Max Pool Layer"<< layerInfo<<"\n";
    }
    //====================================
    out_m = layerInfo["out_Q_m"];
    out_n = layerInfo["out_Q_n"];       
    truncate_m = layerInfo["truncate_m"];
    truncate_n = layerInfo["truncate_n"];;
    activation = layerInfo["activation"];
    use_fixed = layerInfo["use_fixed"];
    truncate_or_not = layerInfo["truncate"];
    layer_name = layerInfo["name"];
    layer_type = MAXPOOL;
    write_output = layerInfo["write_output"];
}