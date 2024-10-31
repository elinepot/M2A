/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x) {
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0) {
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
			(t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {/* x < 0 */
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
			(t * b5 + b4) + b3) + b2) + b1));
	}
}

// Set the state for each thread
__global__ void init_curand_state_k(/*What are the arguments?*/)
{
	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/
}


// Monte Carlo simulation kernel
__global__ void MC_k1(/*What are the arguments?*/){

	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/
}

int main(void) {

	int NTPB = 512;
	int NB = 512;
	int n = NB * NTPB;
	float T = 1.0f;
	float S_0 = 50.0f;
	float K = S_0;
	float sigma = 0.2f;
	float r = 0.1f;
	int N = 100;
	float dt = sqrtf(T/N);
	float sum = 0.0f;
	float sum2 = 0.0f;

	init_curand_state_k<<<NB, NTPB>>>(/*What are the arguments?*/);

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	MC_k1<<<NB, NTPB>>>(/*What are the arguments?*/);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	// Reduction performed on the host
	for (int i = 0; i < n; i++) {
		//sum += ??
		//sum2 += ??
	}

	printf("The estimated price is equal to %f\n", sum);
	printf("error associated to a confidence interval of 95%% = %f\n",
		1.96 * sqrt((double)(1.0f / (n - 1)) * (n*sum2 - (sum * sum)))/sqrt((double)n));
	printf("The true price %f\n", S_0 * NP((r + 0.5 * sigma * sigma)/sigma) -
									K * expf(-r) * NP((r - 0.5 * sigma * sigma) / sigma));
	printf("Execution time %f ms\n", Tim);


	return 0;
}