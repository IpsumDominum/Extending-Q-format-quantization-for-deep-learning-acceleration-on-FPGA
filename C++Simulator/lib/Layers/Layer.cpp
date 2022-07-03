#include "Layers.h"

void Layer::write_output_3D(std::vector<std::vector<std::vector<fixed8>>> output){
    //Write output...
	FILE *fp;
    char* out_directory = (output_dir+"/fixed/"+layer_name+".txt").data();
	fp = fopen(out_directory, "w");
    int output_size_n = output.size();
    int output_size_i = output[0].size();
    int output_size_j = output[0][0].size();
	for (int n = 0; n < output_size_n; n++)
	{
		for (int i = 0; i < output_size_i; i++){
			for (int j = 0; j < output_size_j; j++)
			{
				if (j == output_size_j-1) {
					fprintf(fp, "%f\n", output[n][i][j].fiq());
				}else{
					fprintf(fp, "%f ", output[n][i][j].fiq());
				}
			}
			if (i == output_size_i-1) fprintf(fp, "\n");
		}
			
	}
	fclose(fp);
}
void Layer::write_output_1D(std::vector<fixed8> output){
    //Write output...
	FILE *fp;
    char* out_directory = (output_dir+"/fixed/"+layer_name+".txt").data();
	fp = fopen(out_directory, "w");
	int output_size_n = output.size();
	for (int n = 0; n < output_size_n; n++)
	{
		fprintf(fp, "%f\n", output[n].fiq());
	}
	fclose(fp);
}
//Write double        
void Layer::write_output_3D_double(std::vector<std::vector<std::vector<long double>>> output){
    //Write output...
	FILE *fp;
    char* out_directory = (output_dir+"/double/"+layer_name+"_double.txt").data();
	fp = fopen(out_directory, "w");
    int output_size_n = output.size();
    int output_size_i = output[0].size();
    int output_size_j = output[0][0].size();
	for (int n = 0; n < output_size_n; n++)
	{
		for (int i = 0; i < output_size_i; i++){
			for (int j = 0; j < output_size_j; j++)
			{
				if (j == output_size_j-1) {
					fprintf(fp, "%Lf\n", output[n][i][j]);
				}else{
					fprintf(fp, "%Lf ",output[n][i][j]);
				}
			}
			if (i == output_size_i-1) fprintf(fp, "\n");
		}			
	}
	fclose(fp);
}
void Layer::write_output_1D_double(std::vector<long double> output){
	//Write output...
	FILE *fp;
    char* out_directory = (output_dir+"/double/"+layer_name+"_double.txt").data();
	fp = fopen(out_directory, "w");
	int output_size_n = output.size();
	for (int n = 0; n < output_size_n; n++)
	{
		fprintf(fp, "%Lf\n",output[n]);
	}
	fclose(fp);
}