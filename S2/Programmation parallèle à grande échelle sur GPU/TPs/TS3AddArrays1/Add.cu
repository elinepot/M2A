/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include "timer.h"

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


void addVect(int *a, int *b, int *c, int length){

	int i;

	for(i=0; i<length; i++){
		c[i] = a[i] + b[i];
	}
}


int main (void){

	// Variables definition
	int *a, *b, *c;
	int i;
	
	// Length for the size of arrays
	int length = 1e8;

	Timer Tim;							// CPU timer instructions

	// Memory allocation of arrays 
	a = (int*)malloc(length*sizeof(int));
	b = (int*)malloc(length*sizeof(int));
	c = (int*)malloc(length*sizeof(int));

	// Values initialization
	for(i=0; i<length; i++){
		a[i] = i;
		b[i] = 9*i;
	}

	Tim.start();						// CPU timer instructions

	// Executing the addition 
	addVect(a, b, c, length);

	Tim.add();							// CPU timer instructions

	// Displaying the results to check the correctness 
	for(i=length-50; i<length-45; i++){
		printf(" ( %i ): %i\n", a[i]+b[i], c[i]);
	}

	printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
		   (float)Tim.getsum());		// CPU timer instructions

	// Freeing the memory
	free(a);
	free(b);
	free(c);

	return 0;
}