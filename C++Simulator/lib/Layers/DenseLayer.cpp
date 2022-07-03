#include "Layers.h"
#include "../json.h"
#include <vector>
#include <string>

std::vector<fixed8> DenseLayer::inference1D21D(std::vector<fixed8> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Dense Layer inference"<<"\n";
    }
    std::vector<fixed8> output {};
    fixed8 temp(out_m,out_n);
    //Dense Inference
	for (int i = 0; i < output_nodes_dim; i++)
    {   
		for (int j = 0; j < input_nodes_dim; j++){
			temp = input_matrix[j] * weight[i][j] + temp;
        }
		output.push_back(temp + bias[i]);
        if(truncate_or_not==true){
            output[i] = truncate(output[i], truncate_m, truncate_n);
            if(strcmp(activation.data(),"relu")==0){
                if ((output[i].bin() & 0b1000) != 0)
                    output[i].reset(truncate_m, truncate_n); //Relu
            }                
        }else{
            if(strcmp(activation.data(),"relu")==0){
                if ((output[i].bin() & 0b1000) != 0)
                    output[i].reset(out_m, out_n); //Relu
            }
        }
		temp.reset(out_m,out_n);
	}
    //Inference Complete... Output ready to be retreived
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Output Dimension: "<<"\n";
        std::cout<<"["<<output.size()<<"]"<<"\n";
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

std::vector<long double> DenseLayer::inference1D21D_double(std::vector<long double> input_matrix){
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Dense Layer inference (Double)"<<"\n";
    }    
    std::vector<long double> output {};
    double temp = 0;

    if(strcmp(activation.data(),"relu")==0){
        if(debug_or_not==true){
            std::cout<<"RELU"<<"\n";
        }
    }
    //Dense Inference
	for (int i = 0; i < output_nodes_dim; i++)
    {   
		for (int j = 0; j < input_nodes_dim; j++){
			temp = input_matrix[j] * weight_double[i][j] + temp;
        }
		output.push_back(temp + bias_double[i]);
        if(strcmp(activation.data(),"relu")==0){            
            if (output[i]<0){
                output[i] = 0; //RELU
            }
        }             
		temp = 0;
	}
    //Inference Complete... Output ready to be retreived
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Output Dimension: "<<"\n";
        std::cout<<"["<<output.size()<<"]"<<"\n";
        std::cout<<"================================"<<"\n";
    }
    if(write_output==true){
        write_output_1D_double(output);
    }
    return output;
}

DenseLayer::DenseLayer(json layerInfo,bool debug_set){
    //====================================
    //DEBUG
    if(debug_set==true){
        std::cout<<"DENSE LAYER"<< layerInfo<<"\n";
    }
    debug_or_not = debug_set;
    //====================================
    /*Open File*/
    FILE *w;
    char num[15];
    double temp_float;
    fixed8 temp_fixed;
    weight_m = layerInfo["weight_Q_m"];
    weight_n = layerInfo["weight_Q_n"];
    bias_m = layerInfo["bias_Q_m"];
    bias_n = layerInfo["bias_Q_n"];
    out_m = layerInfo["out_Q_m"];
    out_n = layerInfo["out_Q_n"];   
    input_nodes_dim = layerInfo["input_nodes_dim"];   
    output_nodes_dim = layerInfo["output_nodes_dim"];   
    truncate_m = layerInfo["truncate_m"];
    truncate_n = layerInfo["truncate_n"];;
    activation = layerInfo["activation"];
    use_fixed = layerInfo["use_fixed"];
    truncate_or_not = layerInfo["truncate"];    
    layer_name = layerInfo["name"];
    layer_type = DENSE;
    write_output = layerInfo["write_output"];
    /*
	====================================================================================
	= WEIGHT
	====================================================================================
	*/
    std::string filePathString = layerInfo["weight_file_path"];
    char* filePath = filePathString.data();
    w = fopen(filePath, "r");//file_path[2] = weight1
	if (w != NULL)
	{   
        for (int j = 0; j < output_nodes_dim; j++){
            std::vector<fixed8> v {};
            weight.push_back(v);
            std::vector<long double> v_d {};
            weight_double.push_back(v_d);
			for (int i = 0; i < input_nodes_dim; i++)
			{
                fixed8 temp(weight_m,weight_n);
                double temp_d = 0;
                weight[j].push_back(temp);                
                weight_double[j].push_back(temp_d);
			}
        }
        for (int i = 0; i < input_nodes_dim; i++){
            for (int j = 0; j < output_nodes_dim; j++)
			{
                if (fgets(num, 15, w) != NULL)
				{
					temp_float = atof(num);
					temp_fixed = fq(temp_float, weight_m, weight_n);
                    weight[j][i] = temp_fixed;
                    weight_double[j][i] = temp_float;
				}
			}
        }        
		fclose(w);
	}
	else
	{
		perror( ("Error opening file while loading weight for Dense Layer "+ layer_name + " FilePath:" +filePathString).data());
		exit(0);
	}
    /*
	====================================================================================
	= BIAS
	====================================================================================
	*/
    filePathString = layerInfo["bias_file_path"];
    filePath = filePathString.data();
    w = fopen(filePath, "r");
    if (w != NULL)
    {
        for (int i = 0; i < output_nodes_dim; i++)
		{
			if (fgets(num, 15, w) != NULL)
			{
				temp_float = atof(num);
                temp_fixed = fq(temp_float, bias_m, bias_n);
				bias.push_back(temp_fixed);
                bias_double.push_back(temp_float);
			}
		}
		fclose(w);
    }
    else
    {
        perror( ("Error opening file while loading weight for Dense Layer "+layer_name + " FilePath" + filePathString).data());
        exit(0);
    }
}