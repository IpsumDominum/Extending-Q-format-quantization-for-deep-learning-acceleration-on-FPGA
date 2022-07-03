#include "Network.h"
#include "json.h"
#include <vector>
#include <string>
#define DEBUG

using json = nlohmann::json;

int Network::inference(std::vector<std::vector<fixed8>> input_matrix,std::vector<std::vector<long double>> input_matrix_double){
    std::vector<fixed8> intermediate1D;
    std::vector<std::vector<fixed8>> intermediate2D;
    std::vector<std::vector<std::vector<fixed8>>> intermediate3D = {};

    std::vector<long double> intermediate1D_double;
    std::vector<std::vector<long double>> intermediate2D_double;
    std::vector<std::vector<std::vector<long double>>> intermediate3D_double = {};

    //Unsqueeze one dimension
    intermediate3D.push_back(input_matrix);
    intermediate3D_double.push_back(input_matrix_double);
    for(Layer* layer : layers){
        //DEBUG
        if(debug_or_not==true){
            std::cout<<"Layer: "<<layer->layer_name<<"\n";        
        }
        if(layer->layer_type==CONV){
            if(layer->use_fixed==true){
                intermediate3D = layer->inference3D23D(intermediate3D);
                //Cast in case next layer is double
                intermediate3D_double = cast_fixed_to_double_3D(intermediate3D);
            }else{      
                intermediate3D_double = layer->inference3D23D_double(intermediate3D_double);
                //Cast in case next layer is fixed
                intermediate3D = cast_double_to_fixed_3D(intermediate3D_double,layer->out_m,layer->out_n);
            }
        }else if(layer->layer_type==DENSE){
            if(layer->use_fixed==true){
                intermediate1D = layer->inference1D21D(intermediate1D);
                //Cast in case next layer is double
                intermediate1D_double = cast_fixed_to_double_1D(intermediate1D);
            }else{                
                intermediate1D_double = layer->inference1D21D_double(intermediate1D_double);
                //Cast in case next layer is fixed
                intermediate1D = cast_double_to_fixed_1D(intermediate1D_double,layer->out_m,layer->out_n);
            }
        }else if(layer->layer_type==BATCHNORM_3D){
            if(layer->use_fixed==true){
                intermediate3D = layer->inference3D23D(intermediate3D);
                //Cast in case next layer is double
                intermediate3D_double = cast_fixed_to_double_3D(intermediate3D);
            }else{
                intermediate3D_double = layer->inference3D23D_double(intermediate3D_double);
                //Cast in case next layer is fixed
                intermediate3D = cast_double_to_fixed_3D(intermediate3D_double,layer->out_m,layer->out_n);
            }
        }else if(layer->layer_type==BATCHNORM_1D){
            if(layer->use_fixed==true){
                intermediate1D = layer->inference1D21D(intermediate1D);
                //Cast in case next layer is double
                intermediate1D_double = cast_fixed_to_double_1D(intermediate1D);
            }else{
                intermediate1D_double = layer->inference1D21D_double(intermediate1D_double);
                //Cast in case next layer is fixed
                intermediate1D = cast_double_to_fixed_1D(intermediate1D_double,layer->out_m,layer->out_n);
            }
        }else if(layer->layer_type==FLAT){
            if(layer->use_fixed==true){
                intermediate1D = layer->inference3D21D(intermediate3D);
                //Cast in case next layer is double
                intermediate1D_double = cast_fixed_to_double_1D(intermediate1D);
            }else{
                intermediate1D_double = layer->inference3D21D_double(intermediate3D_double);
                //Cast in case next layer is fixed
                intermediate1D = cast_double_to_fixed_1D(intermediate1D_double,layer->out_m,layer->out_n);
            }
        }else if(layer->layer_type==MAXPOOL){
            if(layer->use_fixed==true){
                intermediate3D = layer->inference3D23D(intermediate3D);
                //Cast in case next layer is double
                intermediate3D_double = cast_fixed_to_double_3D(intermediate3D);
            }else{
                intermediate3D_double = layer->inference3D23D_double(intermediate3D_double);
                //Cast in case next layer is fixed
                intermediate3D = cast_double_to_fixed_3D(intermediate3D_double,layer->out_m,layer->out_n);
            }
        }
    }
    return predict_D(intermediate1D_double);
    //return predict(intermediate1D);
}
int Network::predict_D(std::vector<long double> output)
{
	int ind_max = 0;
    int output_size = output.size();
	for (int i = 0; i < output_size; ++i)
	{
		if(output[i]>output[ind_max]){
			ind_max = i;
		}
	}
    return ind_max;
}

int Network::predict(std::vector<fixed8> network_output)
{
	int indQ = 0;
	uint16_t currentmaxQ = network_output[0].bin();
    int output_size = network_output.size();
	for (int i = 0; i < output_size; i++){
		if (network_output[i].bin() > currentmaxQ)
		{
			currentmaxQ = network_output[i].bin();            
			indQ = i;
		}
        printf("[%d]:%.2f ", i, network_output[i].fiq());
    }
    std::cout<<"\n";
    return indQ;
}

Network::Network(){

}
void Network::addLayer(LayerType layer_type,json layerInfo){
    if(layer_type==CONV){
        layers.push_back(new ConvLayer(layerInfo,debug_or_not));
    }else if(layer_type==BATCHNORM_3D){
        layers.push_back(new BatchNormLayer(layerInfo,"3D",debug_or_not));
    }else if(layer_type==BATCHNORM_1D){
        layers.push_back(new BatchNormLayer(layerInfo,"1D",debug_or_not));
    }else if(layer_type==MAXPOOL){
        layers.push_back(new MaxPoolLayer(layerInfo,debug_or_not));
    }else if(layer_type==FLAT){
        layers.push_back(new FlatLayer(layerInfo,debug_or_not));
    }else if(layer_type==DENSE){
        layers.push_back(new DenseLayer(layerInfo,debug_or_not));
    }
}

LayerType parse_layer_type(std::string type){
    if(type=="conv"){
        return CONV;
    }else if(type=="batch_norm_3D"){
        return BATCHNORM_3D;
    }else if(type=="batch_norm_1D"){
        return BATCHNORM_1D;
    }else if(type=="max_pool"){
        return MAXPOOL;
    }else if(type=="flat"){
        return FLAT;
    }else if(type=="dense"){
        return DENSE;
    }else{
        std::cout<<"Unexpected layer type found in json : "<< type<<"\n";
        exit(EXIT_FAILURE);
    }
}
void Network::build_from_json(std::string json_path){
    // read a JSON file
    std::ifstream i(json_path);
    json j;
    i >> j;
    // write prettified JSON to another file
    // range-based for
    for (json element : j["layers"]) {
        addLayer(parse_layer_type(element["type"]),element);
    }
}
void Network::debug(bool debug_set){
    debug_or_not = debug_set;
}

//Cast fixed -> double
std::vector<std::vector<std::vector<long double>>> Network::cast_fixed_to_double_3D(std::vector<std::vector<std::vector<fixed8>>> to_be_cast){
    int to_be_cast_i = to_be_cast.size();
    int to_be_cast_j = to_be_cast[0].size();
    int to_be_cast_k = to_be_cast[0][0].size();
    std::vector<std::vector<std::vector<long double>>> casted_res = {};
    for(int i=0;i<to_be_cast_i;i++){
        std::vector<std::vector<long double>> v = {};    
        casted_res.push_back(v);
        for(int j=0;j<to_be_cast_j;j++){
            std::vector<long double> vv = {};    
            casted_res[i].push_back(vv);
            for(int k=0;k<to_be_cast_k;k++){
                casted_res[i][j].push_back(to_be_cast[i][j][k].fiq());
            }
        }
    }
    return casted_res;
}
std::vector<long double> Network::cast_fixed_to_double_1D(std::vector<fixed8> to_be_cast){
    int to_be_cast_i = to_be_cast.size();
    std::vector<long double> casted_res = {};
    for(int i=0;i<to_be_cast_i;i++){
        casted_res.push_back(to_be_cast[i].fiq());
    }
    return casted_res;
}
//Cast double -> fixed
std::vector<std::vector<std::vector<fixed8>>> Network::cast_double_to_fixed_3D(std::vector<std::vector<std::vector<long double>>> to_be_cast,int cast_m, int cast_n){
    int to_be_cast_i = to_be_cast.size();
    int to_be_cast_j = to_be_cast[0].size();
    int to_be_cast_k = to_be_cast[0][0].size();
    std::vector<std::vector<std::vector<fixed8>>> casted_res = {};
    for(int i=0;i<to_be_cast_i;i++){
        std::vector<std::vector<fixed8>> v = {};    
        casted_res.push_back(v);
        for(int j=0;j<to_be_cast_j;j++){
            std::vector<fixed8> vv = {};    
            casted_res[i].push_back(vv);
            for(int k=0;k<to_be_cast_k;k++){
                casted_res[i][j].push_back(fq(to_be_cast[i][j][k],cast_m,cast_n));
            }
        }
    }
    return casted_res;
}
std::vector<fixed8> Network::cast_double_to_fixed_1D(std::vector<long double> to_be_cast, int cast_m, int cast_n){
    int to_be_cast_i = to_be_cast.size();
    std::vector<fixed8> casted_res = {};
    for(int i=0;i<to_be_cast_i;i++){
        casted_res.push_back(fq(to_be_cast[i],cast_m,cast_n));
    }
    return casted_res;
}