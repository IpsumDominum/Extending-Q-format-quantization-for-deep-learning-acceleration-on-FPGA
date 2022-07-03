#include "Layers.h"
#include "../json.h"
#include <vector>
#include <string>
#include <cmath>

std::vector<std::vector<std::vector<fixed8>>> BatchNormLayer::inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Batch Norm 3D Layer inference"<<"\n";
    }

    std::vector<std::vector<std::vector<fixed8>>> output {};
    fixed8 temp(out_m,out_n);

    int input_dim = input_matrix.size();
    if(batch_norm_dim != input_dim){
        std::cout<<"Error::Batch Norm Dim does not match Input Dim!"<<"\n";
        std::cout<<"Expected : {"<<batch_norm_dim<<"} Got : {"<<input_dim<<"}\n";
        exit(EXIT_FAILURE);
    }
    //Batch Norm Inference
	for (int i = 0; i < batch_norm_dim;i++) {
        std::vector<std::vector<fixed8>> v;        
        output.push_back(v); 
		for (std::vector<std::vector<fixed8>>::size_type j = 0; j < input_matrix[i].size();j++) {
            std::vector<fixed8> vv;
            output[i].push_back(vv); 
			for (std::vector<fixed8>::size_type k = 0; k < input_matrix[i][j].size();k++) {
				output[i][j].push_back(input_matrix[i][j][k] * batch_inv_std[i]); 
				output[i][j][k] = output[i][j][k] + batch_neg_mean[i];

                if(truncate_or_not==true){
                    output[i][j][k] = truncate(output[i][j][k], truncate_m, truncate_n);
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[i][j][k].bin() & 0b1000) != 0)
                            output[i][j][k].reset(truncate_m, truncate_n); //Relu
                    }                
                }else{
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[i][j][k].bin() & 0b1000) != 0)
                            output[i][j][k].reset(out_m, out_n); //Relu
                    }
                }
			}
		}
	}        
    //Inference Complete... Output ready to be retreived
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Output Shape: ["<<output.size()<<","<<output[0].size()<<","<<output[0][0].size()<<"]"<<"\n";
        std::cout<<"================================"<<"\n";
    }
     if(write_output==true){
        write_output_3D(output);
    }
    return output;
}

std::vector<fixed8> BatchNormLayer::inference1D21D(std::vector<fixed8> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Batch Norm 1D Layer inference"<<"\n";
    }
    std::vector<fixed8> output {};

    int input_dim = input_matrix.size();
    if(batch_norm_dim!=input_dim){
        std::cout<<"Batch Norm 1D dimension does not match input dimension, in layer: "<<layer_name<<"\n";
        std::cout<<"Expected : {"<<batch_norm_dim<<"} Got : {"<<input_dim<<"}\n";
        exit(EXIT_FAILURE);
    }
    // Batch Norm 1D
	for (int i = 0; i < batch_norm_dim; ++i)
	{
		output.push_back(input_matrix[i] * batch_inv_std[i]); // Q(4.0)*Q(2.1)+Q(6.1)
		output[i] = output[i] + batch_neg_mean[i]; // 

         if(truncate_or_not==true){
            output[i]= truncate(output[i], truncate_m, truncate_n);
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
std::vector<std::vector<std::vector<long double>>> BatchNormLayer::inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix){
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Batch Norm 3D Layer inference (Double)"<<"\n";
    }
    std::vector<std::vector<std::vector<long double>>> output {};
    int input_dim = input_matrix.size();
    if(batch_norm_dim != input_dim){
        std::cout<<"Error::Batch Norm Dim does not match Input Dim!"<<"\n";
        std::cout<<"Expected : {"<<batch_norm_dim<<"} Got : {"<<input_dim<<"}\n";
        exit(EXIT_FAILURE);
    }
    if(strcmp(activation.data(),"relu")==0){
        if(debug_or_not==true){
            std::cout<<"RELU"<<"\n";
        }
    }
    //Batch Norm Inference
	for (int i = 0; i < batch_norm_dim;i++) {
        std::vector<std::vector<long double>> v;        
        output.push_back(v); 
		for (std::vector<std::vector<long double>>::size_type j = 0; j < input_matrix[i].size();j++) {
            std::vector<long double> vv;
            output[i].push_back(vv); 
			for (std::vector<long double>::size_type k = 0; k < input_matrix[i][j].size();k++) {
				output[i][j].push_back(input_matrix[i][j][k] * batch_inv_std_double[i]); 
				output[i][j][k] = output[i][j][k] + batch_neg_mean_double[i];
                if(strcmp(activation.data(),"relu")==0){                    
                    if (output[i][j][k]<0){
                        output[i][j][k] = 0.0L; //RELU
                }             
                }
            }
        }        
    }  
    //Inference Complete... Output ready to be retreived
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Output Shape: ["<<output.size()<<","<<output[0].size()<<","<<output[0][0].size()<<"]"<<"\n";
        std::cout<<"================================"<<"\n";
    }
     if(write_output==true){
        write_output_3D_double(output);
    }
    return output;
}

std::vector<long double> BatchNormLayer::inference1D21D_double(std::vector<long double> input_matrix){
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Batch Norm 1D Layer inference (Double)"<<"\n";
    }
    std::vector<long double> output {};
    int input_dim = input_matrix.size();
    if(batch_norm_dim!=input_dim){
        std::cout<<"Batch Norm 1D dimension does not match input dimension, in layer: "<<layer_name<<"\n";
        std::cout<<"Expected : {"<<batch_norm_dim<<"} Got : {"<<input_dim<<"}\n";
        exit(EXIT_FAILURE);
    }
    if(strcmp(activation.data(),"relu")==0){
        if(debug_or_not==true){
            std::cout<<"RELU"<<"\n";
        }
    }
    // Batch Norm 1D
	for (int i = 0; i < batch_norm_dim; ++i)
	{
		output.push_back(input_matrix[i] * batch_inv_std_double[i]); 
		output[i] = output[i] + batch_neg_mean_double[i];
        if(strcmp(activation.data(),"relu")==0){
            if (output[i]<0){
                output[i] = 0.0L; //RELU
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
    if(strcmp(activation.data(),"soft_max")==0){                    
        int output_size = output.size();
        double exp_sum = 0.0;
        for (int i = 0; i < output_size; ++i)
        {
            exp_sum += std::exp(output[i]);
        }
        for (int i = 0; i < output_size; ++i)
        {
            output[i] = std::exp(output[i]) / exp_sum;
        }
    }
    return output;
}


// ===================================================
// LOAD WEIGHT
// ===================================================
BatchNormLayer::BatchNormLayer(json layerInfo,std::string batch_norm_specification,bool debug_set){
    if(strcmp(batch_norm_specification.data(),"3D")==0){
        //====================================
        //DEBUG
        if(debug_set==true){
            std::cout<<"Batch Norm 3D LAYER"<< layerInfo<<"\n";
        }
        //====================================
        layer_type = BATCHNORM_3D;
    }else{
        //====================================
        //DEBUG
        if(debug_set==true){
            std::cout<<"Batch Norm 1D LAYER"<< layerInfo<<"\n";
        }
        //====================================
        layer_type = BATCHNORM_1D;
    }
    debug_or_not = debug_set;
    /*Open File*/
    FILE *w;
    char num[15];
    double temp_float;
    fixed8 temp_fixed;

    if(layerInfo.contains("name")){
        layer_name = layerInfo["name"];
    }else{
        std::string err_msg = "'name' not found in unnamed layer";
        perror(err_msg.data());
        exit(EXIT_FAILURE);
    }

    if(layerInfo.contains("inv_std_Q_m")){
        inv_std_m = layerInfo["inv_std_Q_m"];
    }else{         
        char* err_msg = ("inv_std_Q_m not found in " + layer_name).data();
        perror(err_msg) ;
        exit(EXIT_FAILURE);
    }
    
    if(layerInfo.contains("inv_std_Q_n")){
        inv_std_n = layerInfo["inv_std_Q_n"];   
    }else{
        char* err_msg = ("inv_std_Q_n not found in " + layer_name).data();
        perror(err_msg) ;
        exit(EXIT_FAILURE);
    }

    if(layerInfo.contains("neg_mean_Q_m")){
        neg_mean_m = layerInfo["neg_mean_Q_m"];
    }else{
        char* err_msg = ("neg_mean_Q_m not found in " + layer_name).data();
        perror(err_msg) ;
        exit(EXIT_FAILURE);
    }
    neg_mean_n = layerInfo["neg_mean_Q_n"];   
    out_m = layerInfo["out_Q_m"];
    out_n = layerInfo["out_Q_n"];   
    batch_norm_dim = layerInfo["batch_norm_dim"];   
    truncate_m = layerInfo["truncate_m"];
    truncate_n = layerInfo["truncate_n"];;
    activation = layerInfo["activation"];
    use_fixed = layerInfo["use_fixed"];
    truncate_or_not = layerInfo["truncate"];
    layer_name = layerInfo["name"];
    write_output = layerInfo["write_output"];
    /*
	====================================================================================
	= WEIGHT
	====================================================================================
	*/
    std::string filePathString = layerInfo["inv_std_file_path"];
    char* filePath = filePathString.data();
    w = fopen(filePath, "r");//file_path[2] = weight1
	if (w != NULL)
	{

		for (int i = 0; i < batch_norm_dim; ++i)
		{
			if (fgets(num, 15, w) != NULL)
			{
				temp_float = atof(num);
                temp_fixed = fq(temp_float, inv_std_m,inv_std_n);
				batch_inv_std.push_back(temp_fixed);
                batch_inv_std_double.push_back(temp_float);
			}
		}
		fclose(w);
	}
	else
	{
		perror( ("Error opening file while loading weight for Batch Norm Layer "+ layer_name + " FilePath:" +filePathString).data());
		exit(0);
	}
    /*
	====================================================================================
	= BIAS
	====================================================================================
	*/
    filePathString = layerInfo["neg_mean_file_path"];
    filePath = filePathString.data();
    w = fopen(filePath, "r");
    if (w != NULL)
    {
        for (int i = 0; i < batch_norm_dim; ++i)
		{
			if (fgets(num, 15, w) != NULL)
			{
				temp_float = atof(num);
                temp_fixed = fq(temp_float, neg_mean_m, neg_mean_n);
				batch_neg_mean.push_back(temp_fixed);
                batch_neg_mean_double.push_back(temp_float);
			}
		}
		fclose(w);
    }
    else
    {
        perror( ("Error opening file while loading weight for Batch Norm Layer "+layer_name + " FilePath" + filePathString).data());
        exit(0);
    }
}