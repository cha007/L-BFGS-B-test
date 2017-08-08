#include "mex.h"
#include "ccnf_test.h"

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]){
	callGpuCCNF(nlhs, plhs, nrhs, prhs);
}