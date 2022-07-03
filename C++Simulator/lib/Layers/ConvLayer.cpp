#include "Layers.h"
#include "../json.h"
#include <vector>
#include <string>
#include <float.h>
#include <tgmath.h>


std::vector<std::vector<std::vector<fixed8>>> ConvLayer::inference3D23D(std::vector<std::vector<std::vector<fixed8>>> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Convlayer inference"<<"\n";
    }
    std::vector<std::vector<std::vector<fixed8>>> output {};
    int conv_strides_x = input_matrix[0].size()-conv_kernel_size_x +1;
    int conv_strides_y = input_matrix[0][0].size()-conv_kernel_size_y +1;
    fixed8 temp(out_m,out_n);
	std::vector<std::vector<std::vector<std::vector<fixed8>>>> temp2 {};
	fixed8 temp3(out_m, out_n);
    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Input matrix shape : ["<<input_matrix.size() <<","<<input_matrix[0].size()<<","<<input_matrix[0][0].size()<<"] \n";
        std::cout<<"Conv In : "<<conv_dim_in<<" Conv Out : "<<conv_dim_out<<"\n";
        std::cout<<"Conv strides : ["<<conv_strides_x<<","<<conv_strides_y<<"]\n";
        std::cout<<"Conv kernel : ["<<conv_kernel_size_x<<","<<conv_kernel_size_y<<"]\n";
    }
    //Conv Inference
    int input_matrix_size = input_matrix.size();
    if(input_matrix_size !=conv_dim_in){
        std::cout<< "Conv dimension not equal to input dimension in layer"<< layer_name<<"\n";
        std::cout<< "Expected : "<<conv_dim_in<<",got"<<input_matrix.size()<<"\n";
    }    
    for (int x = 0; x < conv_dim_out; x++){
        std::vector<std::vector<std::vector<fixed8>>> v {};
        temp2.push_back(v);
        for (int i = 0; i < conv_dim_in; i++){
            std::vector<std::vector<fixed8>> vv {};
            temp2[x].push_back(vv);    
            for (int j = 0; j < conv_strides_x; j++){
                std::vector<fixed8> vvv {};
                temp2[x][i].push_back(vvv);    
                for (int k = 0; k < conv_strides_y; k++)
                {   
                    for (int jj = 0; jj < conv_kernel_size_x; jj++)
                        for (int kk = 0; kk < conv_kernel_size_y; kk++)                        
                            temp = weight[x][i][jj][kk] * input_matrix[i][j + jj][k + kk] + temp;
                    temp2[x][i][j].push_back(temp); 
                    temp.reset(out_m, out_n);
                }
            }
        }
    }    
	for (int i = 0; i < conv_dim_out; i++){
        std::vector<std::vector<fixed8>> v {};
        output.push_back(v);
		for (int k = 0; k < conv_strides_x; k++){
            std::vector<fixed8> vv {};
            output[i].push_back(vv);
			for (int l = 0; l < conv_strides_y; l++)
			{
				for (int j = 0; j < conv_dim_in; j++)
					temp3 = temp2[i][j][k][l] + temp3;
				output[i][k].push_back(temp3);
				
				output[i][k][l] = output[i][k][l] + bias[i];

                if(truncate_or_not==true){
                    output[i][k][l] = truncate(output[i][k][l], truncate_m, truncate_n);
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[i][k][l].bin() & 0b1000) != 0)
					    output[i][k][l].reset(truncate_m, truncate_n); //Relu
                    }
                }else{
                    if(strcmp(activation.data(),"relu")==0){
                        if ((output[i][k][l].bin() & 0b1000) != 0)
                            output[i][k][l].reset(out_m, out_n); //Relu
                    }
                }                
				temp3.reset(out_m, out_n);
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

// ===================================================
// DOUBLE INFERENCE
// ===================================================

std::vector<std::vector<std::vector<long double>>> ConvLayer::inference3D23D_double(std::vector<std::vector<std::vector<long double>>> input_matrix) {    
    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Convlayer inference (Double)"<<"\n";
    }    
    std::vector<std::vector<std::vector<long double>>> output {};
    int conv_strides_x = input_matrix[0].size()-conv_kernel_size_x +1;
    int conv_strides_y = input_matrix[0][0].size()-conv_kernel_size_y +1;

    long double temp = 0;
	std::vector<std::vector<std::vector<std::vector<long double>>>> temp2 {};
	long double temp3 = 0;

    //DEBUG
    if(debug_or_not==true){
        std::cout<<"Input matrix shape : ["<<input_matrix.size() <<","<<input_matrix[0].size()<<","<<input_matrix[0][0].size()<<"] \n";
        std::cout<<"Conv In : "<<conv_dim_in<<" Conv Out : "<<conv_dim_out<<"\n";
        std::cout<<"Conv strides : ["<<conv_strides_x<<","<<conv_strides_y<<"]\n";
        std::cout<<"Conv kernel : ["<<conv_kernel_size_x<<","<<conv_kernel_size_y<<"]\n";
    }    

    //Conv Inference
    int input_matrix_size = input_matrix.size();
    if(input_matrix_size !=conv_dim_in){
        std::cout<< "Conv dimension not equal to input dimension in layer"<< layer_name<<"\n";
        std::cout<< "Expected : "<<conv_dim_in<<",got"<<input_matrix.size()<<"\n";
    }    
    for (int x = 0; x < conv_dim_out; x++){
        std::vector<std::vector<std::vector<long double>>> v {};
        temp2.push_back(v);
        for (int i = 0; i < conv_dim_in; i++){
            std::vector<std::vector<long double>> vv {};
            temp2[x].push_back(vv);    
            for (int j = 0; j < conv_strides_x; j++){
                std::vector<long double> vvv {};
                temp2[x][i].push_back(vvv);    
                for (int k = 0; k < conv_strides_y; k++)
                {   
                    for (int jj = 0; jj < conv_kernel_size_x; jj++)
                        for (int kk = 0; kk < conv_kernel_size_y; kk++){
                            temp = weight_double[x][i][jj][kk] * input_matrix[i][j + jj][k + kk] + temp;
                        }
                    temp2[x][i][j].push_back(temp);                     
                    temp = 0;
                }
            }
        }
    }    
    if(debug_or_not==true){
        std::cout<<"Adding Bias"<<"\n";
    }
    if(strcmp(activation.data(),"relu")==0){
        if(debug_or_not==true){
            std::cout<<"RELU"<<"\n";
        }
    }
	for (int i = 0; i < conv_dim_out; i++){
        std::vector<std::vector<long double>> v {};
        output.push_back(v);
		for (int k = 0; k < conv_strides_x; k++){
            std::vector<long double> vv {};
            output[i].push_back(vv);
			for (int l = 0; l < conv_strides_y; l++)
			{
				for (int j = 0; j < conv_dim_in; j++)
					temp3 = temp2[i][j][k][l] + temp3;
				output[i][k].push_back(temp3);				
				output[i][k][l] = output[i][k][l] + bias_double[i];

                if(strcmp(activation.data(),"relu")==0){                    
                    if (output[i][k][l]<0){
                        output[i][k][l] = 0.0L; //RELU
                    }             
                }
				temp3 = 0;
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



ConvLayer::ConvLayer(json layerInfo,bool debug_set){
    //====================================
    //DEBUG
    if(debug_set==true){
        std::cout<<"CONV LAYER"<< layerInfo<<"\n";        
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
    truncate_m = layerInfo["truncate_m"];
    truncate_n = layerInfo["truncate_n"];
    activation = layerInfo["activation"];
    use_fixed = layerInfo["use_fixed"];
    truncate_or_not = layerInfo["truncate"];
    conv_dim_in = layerInfo["conv_dim_in"];
    conv_dim_out = layerInfo["conv_dim_out"];
    conv_kernel_size_x = layerInfo["conv_kernel_size_x"];
    conv_kernel_size_y = layerInfo["conv_kernel_size_y"];
    layer_name = layerInfo["name"];    
    layer_type = CONV;
    write_output = layerInfo["write_output"];
    /*
	====================================================================================
	= WEIGHT
	====================================================================================
	*/
    std::string filePathString = layerInfo["weight_file_path"];
    char* filePath = filePathString.data();
    w = fopen(filePath, "r");
	if (w != NULL)
	{   
        /*Load Weight*/
        for(int x=0;x<conv_dim_out;x++){
            std::vector<std::vector<std::vector<fixed8>>> v  {};
            weight.push_back(v);            
            std::vector<std::vector<std::vector<long double>>> v_d  {};
            weight_double.push_back(v_d);            
            for(int i=0;i<conv_dim_in;i++){
                std::vector<std::vector<fixed8>> vv  {};
                weight[x].push_back(vv);            
                std::vector<std::vector<long double>> vv_d  {};
                weight_double[x].push_back(vv_d);            
                for(int i_x=0;i_x< conv_kernel_size_x;i_x++){
                    std::vector<fixed8> vvv {};
                    weight[x][i].push_back(vvv);
                    std::vector<long double> vvv_d {};
                    weight_double[x][i].push_back(vvv_d);
                    for(int i_y=0;i_y< conv_kernel_size_y;i_y++){
                        //Load weight according to 
                        //weight size specifications
                        //and Q dimension
                        if (fgets(num, 15, w) != NULL)
                        {
                            temp_float = atof(num);//convert num into double
                            temp_fixed = fq(temp_float,weight_m,weight_n);
                            weight[x][i][i_x].push_back(temp_fixed);
                            weight_double[x][i][i_x].push_back(temp_float);
                        }
                    }
                }
            }
        }
        //Close File
        fclose(w);
	}
	else
	{
		perror( ("Error opening file while loading weight for Conv Layer " + layer_name + " FilePath:" + filePathString).data());
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
        for (int i = 0; i < conv_dim_out; ++i)
        {
            if (fgets(num, 15, w) != NULL)
            {
                temp_float = atof(num);                
                temp_fixed = fq(temp_float, bias_m,bias_n);
                bias.push_back(temp_fixed);
                bias_double.push_back(temp_float);
            }
        }
        fclose(w);
    }
    else
    {
        perror( ("Error opening file while loading bias for Conv Layer " + layer_name + " FilePath:" + filePathString).data());
        exit(0);
    }
}