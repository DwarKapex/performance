#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "sys/stat.h"
#include <sstream>
#include <thread>
#include <immintrin.h>

#include "datetime.h"
#include "allocation.h"
#include "inner_product.h"
#include "train.h"
#include "reader.h"
#include "activation_functions.h"
#include "model.h"

#define ALIGN_ALLOC(align, size) aligned_alloc(align, size)
#define ALIGN_FREE(memory) free(memory)




/* Use in aNN-class for allocation memory*/
#include <unistd.h>
int is_avx_supported()
{
    unsigned int eax, ebx, ecx, edx;
    cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 28)) ? 1:0;
}
long getCacheLineSize()
{
    long l1dcls = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
       if (l1dcls == -1)
        l1dcls = sizeof(void*);
    return l1dcls;
}


bool allocation_1D (float *&data, size_t const size){
    long l1dcls = getCacheLineSize();
    if (!(data = (float*)ALIGN_ALLOC(l1dcls, size * sizeof(float)))){
        std::cout << "1D data allocation error!\n";
        return false;
    }
    return true;
}




void randomInitMatrix(float ** matrix, size_t const len, size_t const size)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(-0.05f, 0.05f);

    for (int row = 0; row < len; ++row)
        for (int col = 0; col < size; ++col)
            matrix[row][col] = (float)distribution(generator);
}

void randomInitVec(float * vec, size_t const len)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(-0.05f, 0.05f);

        for (int col = 0; col < len; ++col)
            vec[col] = distribution(generator);
}

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}


#ifdef __AVX2__
// res += scalar*inVec
inline void scalarByVecProd(float * __restrict result, float const scalar, float const * __restrict inVec, size_t const vecSize)
{
    __m256 *res = (__m256*) result;
    __m256 *in  = (__m256*) inVec;
    
    __m256 sc, s;
    sc = _mm256_broadcast_ss(&scalar);
    for (int i = 0; i < vecSize/8; ++i){
        s = _mm256_mul_ps(sc, in[i]);
        res[i] = _mm256_add_ps (res[i], s);
    }
}

float getOutputValue(float *inputVec, float *outputLayer, size_t hdSize)
{

    __m256 *xx = (__m256*)inputVec;
    __m256 *yy = (__m256*)outputLayer;
    __m256 s, p;
    s = _mm256_setzero_ps();
    
    for(int i = 0; i < hdSize / 8; ++i){
        p = _mm256_dp_ps (xx[i], yy[i], 0xFF);
        s = _mm256_add_ps(s,p);
    }
    
    p =_mm256_permute2f128_ps (s, s, 1);
    s = _mm256_add_ps(s,p);
    return _mm256_get_first(s);
}
void inline inner_avx256(float * __restrict val, int const n, float const * __restrict x, float const * __restrict y){

    __m256 *xx = (__m256*)x;
    __m256 *yy = (__m256*)y;
    __m256 s, p, v;
    s = _mm256_setzero_ps();
    
    v = _mm256_broadcast_ss(val);
    for(int i = 0; i < n/8; ++i){
        p = _mm256_dp_ps (xx[i], yy[i], 0xFF);
        s = _mm256_add_ps(s,p);
    }
    
    p =_mm256_permute2f128_ps (s, s, 1);
    s = _mm256_add_ps(s,p);
    s = _mm256_add_ps(s,v); 
    *val = _mm256_get_first(s);
}

void innerProd( float * __restrict result, float const * __restrict inVec, float const ** __restrict matrix, 
                       size_t const inVecSize, size_t const matrixSize )
{
    for (int row = 0; row < matrixSize; ++row)
        inner_avx256((result+row), (int)inVecSize, inVec, matrix[row]);
}

#else

float getOutputValue(float *inputVec, float const *outputLayer, size_t const hdSize)
{
    float res = 0.0f;
    for (int row = 0; row < hdSize; row++)
        res += inputVec[row]*outputLayer[row];
    return res;
}

void innerProd( float * __restrict result, float const * __restrict inVec, float const ** __restrict matrix, 
                       size_t const inVecSize, size_t const matrixSize )
{
    float value = 0.0f;
    for (int row = 0; row < matrixSize; ++row){
        value = result[row];
        for (int col = 0; col < inVecSize; ++col)
            value += inVec[col]*matrix[row][col];
        result[row] = value;
    }
}

#endif // __AVX2__

void add_bias(float ** pipe, float * biasInputLayer, float ** biasHiddenLayers,
              size_t const nnDepth, size_t const hiddenLayerSize)
{
    
    memcpy(pipe[0], biasInputLayer, hiddenLayerSize*sizeof(float));
    
    for (size_t row = 0; row < nnDepth-1; ++row)
        memcpy(pipe[row+1], biasHiddenLayers[row], hiddenLayerSize*sizeof(float));

    for (size_t row = nnDepth; row < 2 * nnDepth; ++row)
        memset(pipe[row], 0, hiddenLayerSize*sizeof(float));
}

void hiddenLayerForwardProp(float ** pipe, float const *** nnHiddenLayers, 
                            size_t const nnDepth, size_t const hiddenLayerSize)
{
	for (size_t ihiddenLayer = 0; ihiddenLayer < nnDepth - 1; ++ihiddenLayer)
            innerProd(pipe[ihiddenLayer + 1], pipe[ihiddenLayer],
                      nnHiddenLayers[ihiddenLayer], hiddenLayerSize, hiddenLayerSize);
}

void train( float ** data, int inputVecSize, int dataLength,
            float ** target,
            aNN& model, float **pipe,
            aNNUpdate& update, float learningRate,
	    float * error, int iThread, int maxThreads)
{
    float outputValue = 0.0f,
          error = 0.0f;
    
    size_t  dataPieceLen = dataLength / maxThreads + 1,
                   start = dataPieceLen * iThread,
                   end = (dataLength < start + dataPieceLen ? dataLength : start + dataPieceLen);
             
    for (size_t iData = start; iData < end; ++iData){
        
        add_bias(pipe, model);
        
        /* Step 1. Convert current data vector with input layer*/
        inputLayerForwardProp(pipe[0], data[iData], model);
        
        /* Step 2. Propagate result through hidden layers*/
        hiddenLayerForwardProp(pipe, model);
        
        /* Get output error for further back propagation */
	outputValue = getOutputValue(pipe[nnDepth-1], model);

        error = ( target[0][iData] - outputValue );
        *totalError += error*error;
        
        /* Step 3. Back propogate with output layer */
	outputLayerBackProp(pipe[nnDepth], error, pipe[nnDepth - 1],
			     model, update, learningRate);
        
        /* Step 4. Back propogate with hidden layers */
	hiddenLayerBackProp(pipe, model, update, learningRate);
        
	/* Step 5. Back propogate with input layers */
        inputLayerBackProp( pipe[2*nnDepth-1], model, update,
                            data[iData], inputVecSize, learningRate);
    }    
}




/* ======================================================= */

int main(void)
{
	
    int       inputDataSize = 0;				/* the number of input vectors */
    const int inputVecSize =  40,				/* size of input vector */
              
              NeuralNetworkDepth = 8,				/* number of hidden layers of Neural Network */
              hiddenLayerSize = 16,  				/* size of hidden layer matrix */
              maxThreads = 1,                           	/* max number of threads to be in use*/
              maxEpochs = 200;                                  /* max number of epochs during training */

    double   ** data = nullptr,					/* pointer to data array*/
             ** target = nullptr;				/* pointer to target array*/

	
    maxThreads = std::thread::hardware_concurrency();

    std:string source = "../data/input.data";

	inputDataSize = readData(source, target, data, inputVecSize);
	normalizeData(data, target, inputVecSize, inputDataSize);

	aNN model (inputVecSize, NeuralNetworkDepth, hiddenLayerSize);
	std::vector<aNNUpdate> updates(maxThreads, aNNUpdate(inputVecSize, NeuralNetworkDepth, hiddenLayerSize));

	std::stringstream a, b;
    	a << "../report/report_" << "vec" << inputVecSize << "_NN_" << NeuralNetworkDepth << "_" 
          << hiddenLayerSize << "x" << hiddenLayerSize << "_lr" << learningRate << "_nTh" 
	  << maxThreads;

#ifdef __AVX2__
    a << "_avx2";
#endif
    std::string begin = a.str();
    int i = 0;
    do {
        b.str("");
        b << begin;
        b << "[" << i << "].txt";
        ++i;
    } while ( file_exists(b.str()) );
        
    std::string outputFileName = b.str(); 
    std::cout << "output file name = " << outputFileName << std::endl;
    std::ofstream outputFile;

    std::vector<std::thread> workers(maxThreads);

    std::vector<double> error = {};
	
    for (int iEpochs = 0; iEpochs < maxEpochs; ++iEpochs){
        
        std::cout << "Epoch " << iEpochs << " ";
	long long startStamp = StampNow();
            
        
        // zero error vector
        std::fill(error.begin(), error.end(), 0);
        

	for (int t = 0; t < maxThreads; t++){ 
            workers[t] = std::thread(train, data, inputVecSize, inputDataSize,
                                     target, std::ref(model),
                        	     pipeline[t],
                          	     std::ref(updates[t]),	
                                     learningRate, error + t, t, maxThreads);
        
	// cpu afinity
	cpu_set_t cpuset;
    	CPU_ZERO(&cpuset);
    	CPU_SET(t, &cpuset);
    	int rc = pthread_setaffinity_np(workers[t].native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
   	 if (rc != 0) 
      		std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
   	 
	}
        for (int t = 0; t < maxThreads; t++) 
		if (workers[t].joinable())
			workers[t].join();
        
        weightsUpdate( model, updates );  
        
        double totalError = 0.0f;
        for (int iThread = 0; iThread < maxThreads; ++iThread){
            totalError += error[iThread];
        }
	float timeSpent = float(StampNow() - startStamp) / TICKS_PER_SECOND;
        std::cout << totalError << " (spent " << timeSpent << " s) " << std::endl;

        saveReport( outputFile, model, update, totalError, timeSpent);
 
    }
    
    saveNeuralNetwork("NN_sample.data", model);

    return 0;
}








