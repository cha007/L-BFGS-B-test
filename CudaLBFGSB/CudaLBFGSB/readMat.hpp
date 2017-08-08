#pragma once
#include "mat.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#pragma comment(lib, "libmat.lib")
#pragma comment(lib,"libmx.lib")

typedef struct MatlabVec{
	std::vector<double> data;
	int M;
	int N;
}MatlabVec;

void matread(const char *file, const char *name, MatlabVec& v){
	// open MAT-file
	MATFile *pmat = matOpen(file, "r");
	if (pmat == NULL) return;

	// extract the specified variable
	mxArray *arr = matGetVariable(pmat, name);
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
		// copy data
		mwSize num = mxGetNumberOfElements(arr);
		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			v.data.resize(num);
			v.data.assign(pr, pr + num);
		}
		v.M = mxGetM(arr);
		v.N = mxGetN(arr);
		printf("read mat %s [%d %d]\n", name, v.M, v.N);
	}
	else{
		printf("%s error\n", name);
	}

	// cleanup
	mxDestroyArray(arr);
	matClose(pmat);
}

void matRead(const mxArray *arr, MatlabVec& v){
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
		// copy data
		mwSize num = mxGetNumberOfElements(arr);
		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			v.data.resize(num);
			v.data.assign(pr, pr + num);
		}
		v.M = mxGetM(arr);
		v.N = mxGetN(arr);
		printf("read mat [%d %d]\n", v.M, v.N);
	}
	else{
		printf(" error\n");
	}
}