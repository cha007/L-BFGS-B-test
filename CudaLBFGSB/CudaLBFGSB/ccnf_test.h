#pragma once
#include "mat.h"

int callCpuCCNF();
int callCpuCCNF_test();
void callGpuCCNF(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);