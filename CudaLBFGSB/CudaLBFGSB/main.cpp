#include <stdio.h>
#include "ccnf_test.h"
#include "L-BFGS/lbfgsbcuda.h"
#include "L-BFGS/lbfgsb.h"
extern real num_size;

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream){

}
int main()
{
	callCpuCCNF();
	printf("hello\n");
	getchar();
	return 0;
}