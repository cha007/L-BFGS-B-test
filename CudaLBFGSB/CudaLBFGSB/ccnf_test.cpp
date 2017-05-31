#include "ccnf_test.h"
#include "readMat.hpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "L-BFGS/lbfgsbcuda.h"
#include "L-BFGS/lbfgsb.h"
#include <fstream> 

using namespace std;
using namespace Eigen;
cublasHandle_t cublasHd;
double stpscal;
int num_params;
void randInitializeWeights(MatrixXd* initial_Theta, int L_in, int L_out){
	double epsilon_init = 1.0f / sqrtf(L_in);
	for (int i = 0; i < initial_Theta->rows(); ++i) {
		for (int j = 0; j < initial_Theta->cols(); ++j) {
			double val = (double)rand() / (double)RAND_MAX;
			initial_Theta->coeffRef(i, j) = val * 2.0 * epsilon_init - epsilon_init;			
		}
	}
}

void CalcSigmaCCNFflat(
	VectorXd* alphas, 
	VectorXd* betas,
	int seq_length,
	MatrixXd* precalc_B_without_beta,
	MatrixXd* precalc_eye,
	MatrixXd* precalc_zeros,
	MatrixXd* SigmaInv){
	MatrixXd A = alphas->sum() * *precalc_eye;
	MatrixXd Btmp = *precalc_B_without_beta * *betas;
	//cout << (*precalc_B_without_beta) << endl;
	//cout << Btmp << endl;
	MatrixXd B = *precalc_zeros;

	int count = 0;
	for (int i = 0; i < B.rows(); ++i){
		count = 0;
		for (int j = 0; j < B.cols(); ++j) {
			if (i >= j){
				count += j;
				int bb = B.cols() * j + i - count;
				double aa = Btmp.coeffRef(bb);
				B.coeffRef(i, j) = aa;
			}
		}
	}
	MatrixXd B_ = B.adjoint();
	for (int i = 0; i < B.rows(); ++i){
		count = 0;
		for (int j = 0; j < B.cols(); ++j) {
			if (i >= j){
				count += j;
				int bb = B.cols() * j + i - count;
				double aa = Btmp.coeffRef(bb);
				B_.coeffRef(i, j) = aa;
			}
		}
	}
	(*SigmaInv) = 2.0 * (A + B_);
}

float LogLikelihoodCCNF(
	MatrixXd* ys, 
	MatrixXd* xs, 
	VectorXd* alpha, 
	VectorXd* betas,
	MatrixXd* thetas,
	float lambda_a,
	float lambda_b, 
	float lambda_th,
	MatrixXd* Precalc_Bs_flat,
	MatrixXd* SigmaInvs,
	MatrixXd* ChDecomps, 
	MatrixXd* Sigmas,
	MatrixXd* bs, 
	int num_seq, 
	MatrixXd* all_X_resp){
	int seq_length = ys->rows();
	int num_seqs = ys->cols();
	MatrixXd x_resp = MatrixXd::Zero(thetas->rows(), xs->cols());
	if (SigmaInvs == NULL){
		MatrixXd a = (*thetas * *xs);
		MatrixXd b = a.array().exp();
		x_resp = 1.0 + b.array();
	}
	MatrixXd all_bs = 2 * alpha->adjoint() * x_resp.cwiseInverse();
	MatrixXd precalc_eye = MatrixXd::Identity(seq_length, seq_length);
	MatrixXd precalc_zeros = MatrixXd::Zero(seq_length, seq_length);
	MatrixXd SigmaInv = MatrixXd::Zero(seq_length, seq_length);
	CalcSigmaCCNFflat(
		alpha,
		betas,
		seq_length,
		Precalc_Bs_flat,
		&precalc_eye,
		&precalc_zeros,
		&SigmaInv);
	LLT<MatrixXd> lltOfA(SigmaInv); // compute the Cholesky decomposition of A
	MatrixXd L = lltOfA.matrixL();
	MatrixXd CholDecomp = L.adjoint();
	MatrixXd aa = L.inverse() * precalc_eye;
	MatrixXd Sigma = CholDecomp.inverse() * aa;
	MatrixXd all_bs_reshape = Map<MatrixXd>(all_bs.data(), seq_length, num_seqs);
	MatrixXd mus = Sigma * all_bs_reshape;
	double log_normalisation = num_seqs * CholDecomp.diagonal().array().log().sum();
	MatrixXd ymu = *ys - mus;
	MatrixXd y1 = SigmaInv * ymu;
	MatrixXd ymu_reshape = Map<MatrixXd>(ymu.data(), seq_length * num_seqs, 1);
	MatrixXd y1_reshape = Map<MatrixXd>(y1.data(), seq_length * num_seqs, 1);
	MatrixXd aaa = ymu_reshape.adjoint() * y1_reshape;
	double logL = log_normalisation - 0.5 *  aaa.coeffRef(0,0);
	MatrixXd temp_betas = (*betas).adjoint() * (*betas);
	MatrixXd temp_alphas = (*alpha).adjoint() * (*alpha);
	MatrixXd thetas_remap = Map<MatrixXd>((*thetas).data(), (*thetas).rows() * (*thetas).cols(), 1);
	MatrixXd temp_thetas_remap = thetas_remap.adjoint() * thetas_remap;
	logL = logL - lambda_b * temp_betas.coeffRef(0, 0) / 2 
		- lambda_a * temp_alphas.coeffRef(0, 0) / 2 
		- lambda_th * temp_thetas_remap.coeffRef(0, 0) / 2;
	return logL;
}
ofstream out("out.txt");
void objectiveFunction(
	double *loss, 
	VectorXd *gradient,
	VectorXd *params,
	int numAlpha, 
	int numBeta,
	int sizeTheta_x,
	int sizeTheta_y,
	float lambda_a,
	float lambda_b,
	float lambda_th,
	MatrixXd* PrecalcQ2s0,
	MatrixXd* PrecalcQ2s1,
	MatrixXd* PrecalcQ2s2,
	MatrixXd* x,
	MatrixXd* y,
	MatrixXd* PrecalcYqDs,
	MatrixXd* PrecalcQ2sFlat){

	VectorXd alpha = params->head(numAlpha);
	VectorXd betas = params->segment(numAlpha, numBeta);
	MatrixXd thetas(Map<MatrixXd>(params->tail(sizeTheta_x * sizeTheta_y).data(),
		sizeTheta_x, sizeTheta_y));
	out << "*params" << endl << params->adjoint() << endl;
	int num_seqs = PrecalcYqDs->rows();
	//gradientCCNF
	MatrixXd allXresp = MatrixXd::Zero(thetas.rows(), x->cols());
	MatrixXd a = (-thetas * *x);
	MatrixXd b = a.array().exp();
	allXresp = (1.0 + b.array()).cwiseInverse();
 //out << "allXresp" << endl << allXresp << endl;
	MatrixXd Xt = *x;
	MatrixXd all_bs = 2 * alpha.adjoint() * allXresp;
 //out << "all_bs" << endl << all_bs << endl;
	MatrixXd db2_precalc = MatrixXd::Zero(thetas.rows(), x->cols());
	int num_feats = thetas.cols();
	int seq_length = x->cols() / num_seqs;
	VectorXd gradientParams = VectorXd::Zero(alpha.size() + betas.size() + thetas.rows() * thetas.cols());
	MatrixXd SigmaInv = MatrixXd::Zero(seq_length, seq_length);
	MatrixXd CholDecomp;		MatrixXd Sigma;		MatrixXd mus_remap;		MatrixXd mus;
	if (1){
		MatrixXd I = MatrixXd::Identity(seq_length, seq_length);
		MatrixXd precalc_zeros = MatrixXd::Zero(seq_length, seq_length);

		CalcSigmaCCNFflat(
			&alpha,
			&betas,
			seq_length,
			PrecalcQ2sFlat,
			&I,
			&precalc_zeros,
			&SigmaInv);
	 //out << "SigmaInv" << endl << SigmaInv << endl;
		LLT<MatrixXd> lltOfA(SigmaInv); // compute the Cholesky decomposition of A
		MatrixXd L = lltOfA.matrixL();
		CholDecomp = L.adjoint();
	 //out << "CholDecomp" << endl << CholDecomp << endl;
		MatrixXd aa = L.inverse() * I;
		Sigma = CholDecomp.inverse() * aa;
	 //out << "Sigma" << endl << Sigma << endl;
		double Sigma_trace = Sigma.trace();
		MatrixXd all_bs_reshape = Map<MatrixXd>(all_bs.data(), seq_length, num_seqs);
		mus = Sigma * all_bs_reshape;
	 //out << "mus" << endl << mus << endl;
		MatrixXd diff = *y - mus;
	 //out << "diff" << endl << diff << endl;
		VectorXd B(Map<VectorXd>(diff.data(), diff.cols()*diff.rows()));
		for (int i = 0; i < allXresp.rows(); ++i){
			for (int j = 0; j < allXresp.cols(); ++j) {
				db2_precalc.coeffRef(i, j) =
					2.0 * (1.0 - allXresp.coeffRef(i, j)) * allXresp.coeffRef(i, j) * alpha.coeffRef(i) *B.coeffRef(j);
			}
		}
	 //out << "db2_precalc" << endl << db2_precalc << endl;
		MatrixXd gradientThetasT = Xt * db2_precalc.adjoint();
		double gradient_alphas_add = -y->squaredNorm() + mus.squaredNorm() + num_seqs * Sigma_trace;
		MatrixXd y_remap = Map<MatrixXd>((*y).data(), (*y).rows() * (*y).cols(), 1);
		mus_remap = Map<MatrixXd>(mus.data(), mus.rows() * mus.cols(), 1);
		VectorXd gradient_alphas_add_vec = VectorXd::Constant(numAlpha, gradient_alphas_add);
		VectorXd gradient_alphas = 2 * allXresp * (y_remap - mus_remap) + gradient_alphas_add_vec;
		VectorXd gradient_betas = VectorXd::Zero(numBeta);

		MatrixXd B_k_mu[3];
		B_k_mu[0] = (*PrecalcQ2s0) * mus;
		B_k_mu[1] = (*PrecalcQ2s1) * mus;
		B_k_mu[2] = (*PrecalcQ2s2) * mus;
		MatrixXd Sigma_remap = Map<MatrixXd>(Sigma.data(), Sigma.rows() * Sigma.cols(), 1);
		MatrixXd B_k_remap[3];
		B_k_remap[0] = Map<MatrixXd>((*PrecalcQ2s0).data(), (*PrecalcQ2s0).rows() * (*PrecalcQ2s0).cols(), 1);
		B_k_remap[1] = Map<MatrixXd>((*PrecalcQ2s1).data(), (*PrecalcQ2s1).rows() * (*PrecalcQ2s1).cols(), 1);
		B_k_remap[2] = Map<MatrixXd>((*PrecalcQ2s2).data(), (*PrecalcQ2s2).rows() * (*PrecalcQ2s2).cols(), 1);
		for (int i = 0; i < numBeta; i++){
			double yq_B_k_yq = PrecalcYqDs->col(i).sum();
			MatrixXd B_k_mu_remap = Map<MatrixXd>(B_k_mu[i].data(), B_k_mu[i].rows() * B_k_mu[i].cols(), 1);
			MatrixXd mu_B_k_mu = mus_remap.adjoint() * B_k_mu_remap;
			MatrixXd partition_term = num_seqs * Sigma_remap.adjoint() * B_k_remap[i];
			double dLdb = yq_B_k_yq + mu_B_k_mu.coeffRef(0, 0) + partition_term.coeffRef(0, 0);
			gradient_betas[i] = dLdb;
		}
	 //out << "gradient_alphas" << endl << gradient_alphas << endl;
	 //out << "gradient_betas" << endl << gradient_betas << endl;
	 //out << "gradientThetasT" << endl << gradientThetasT << endl;
		MatrixXd gradientThetas = gradientThetasT.adjoint();
		VectorXd BB(Map<VectorXd>(gradientThetas.data(), gradientThetasT.cols()*gradientThetasT.rows()));
		gradientParams << gradient_alphas, gradient_betas, BB;
	 //out << "gradientParams" << endl << gradientParams << endl;
// 		cout << gradient_alphas;
// 		cout << gradient_betas;
// 		cout << gradientParams;

	}

	VectorXd regAlpha = alpha * lambda_a;
	VectorXd regBeta = betas * lambda_b;
	MatrixXd regTheta = thetas * lambda_th;
	VectorXd gradientParamsSub = VectorXd::Zero(alpha.size() + betas.size() + thetas.rows() * thetas.cols());
	VectorXd BBBB(Map<VectorXd>(regTheta.data(), regTheta.cols()*regTheta.rows()));
	gradientParamsSub << regAlpha, regBeta, BBBB;
 //out << "gradientParamsSub" << endl << gradientParamsSub << endl;
	gradientParams -= gradientParamsSub;
	*gradient = -gradientParams;
 out << "*gradient" << endl << (*gradient).adjoint() << endl;
 //out << "*CholDecomp" << endl << CholDecomp << endl;
	double log_normalisation = num_seqs * CholDecomp.diagonal().array().log().sum();
	MatrixXd ymu = *y - mus;
	MatrixXd y1 = SigmaInv * ymu;
	MatrixXd ymu_reshape = Map<MatrixXd>(ymu.data(), seq_length * num_seqs, 1);
	MatrixXd y1_reshape = Map<MatrixXd>(y1.data(), seq_length * num_seqs, 1);
	MatrixXd aaa = ymu_reshape.adjoint() * y1_reshape;
	double logL = log_normalisation - 0.5 *  aaa.coeffRef(0, 0);
	MatrixXd temp_betas = betas.adjoint() * betas;
	MatrixXd temp_alphas = alpha.adjoint() * alpha;
	MatrixXd thetas_remap = Map<MatrixXd>(thetas.data(), thetas.rows() * thetas.cols(), 1);
	MatrixXd temp_thetas_remap = thetas_remap.adjoint() * thetas_remap;
	logL = logL - lambda_b * temp_betas.coeffRef(0, 0) / 2
		- lambda_a * temp_alphas.coeffRef(0, 0) / 2
		- lambda_th * temp_thetas_remap.coeffRef(0, 0) / 2;
	*loss = -1 * logL;
	out << "*loss" << endl << *loss << endl;
}


int max_iter = 200;
int input_layer_size = 121;
int patch_length = 81;

int best_num_layer = 7;
float best_lambda_a = 200.0f;
float best_lambda_b = 7500.0f;
float best_lambda_th = 1.0f;
int num_seqs = 2915;
int num_reinit = 20;
realreal* xx_opti;
MatrixXd xEigen;
MatrixXd yEigen;
MatrixXd Precalc_Bs_0_Eigen;
MatrixXd Precalc_Bs_1_Eigen;
MatrixXd Precalc_Bs_2_Eigen;
MatrixXd Precalc_Bs_flatEigen;
MatrixXd Precalc_yBysEigen;
/*

class cpu_ccnf : public cpu_cost_function
{
public:
	cpu_ccnf(size_t n)
		: cpu_cost_function(n) {

	}

	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		*h_y = 0.0f;

		for (size_t i = 0; i < m_numDimensions / 2; ++i)
		{
			const floatdouble x0 = h_x[2 * i + 0];
			const floatdouble x1 = h_x[2 * i + 1];

			// f = (1-x0)^2 + 100 (x1-x0^2)^2

			const floatdouble a = (1.0 - x0);
			const floatdouble b = (x1 - x0 * x0);

			*h_y += (a*a) + 100.0f * (b*b);
		}
	}

	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		for (size_t i = 0; i < m_numDimensions / 2; ++i)
		{
			const floatdouble x0 = h_x[2 * i + 0];
			const floatdouble x1 = h_x[2 * i + 1];

			// df/dx0 = -2 (1-x0) - 400 (x1-x0^2) x0
			// df/dx1 = 200 (x1 - x0^2)

			h_grad[2 * i + 0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
			h_grad[2 * i + 1] = 200.0f * (x1 - x0*x0);
		}
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
// 		cpu_f(h_x, h_f);
// 		cpu_gradf(h_x, h_gradf);
		static int count_ = 0;
		cout << "[" << count_ << "] begin" << endl;
		VectorXf params_f = Map<Eigen::VectorXf>((float*)h_x, m_numDimensions, 1);
		VectorXd params = params_f.cast <double>();
		double loss;
		VectorXd gradient = VectorXd::Zero(m_numDimensions);
		objectiveFunction(
			&loss,
			&gradient,
			&params,
			best_num_layer,
			3,
			best_num_layer,
			(input_layer_size + 1),
			best_lambda_a,
			best_lambda_b,
			best_lambda_th,
			&Precalc_Bs_0_Eigen,
			&Precalc_Bs_1_Eigen,
			&Precalc_Bs_2_Eigen,
			&xEigen,
			&yEigen,
			&Precalc_yBysEigen,
			&Precalc_Bs_flatEigen);
		*h_f = (float)loss;

		cout << "[" << count_++ << "] " << *h_f << endl;
		VectorXf params_ff = gradient.cast <float>();
		for (size_t i = 0; i < m_numDimensions ; ++i){
			h_gradf[i] = params_ff[i];
		}
	}
};
*/

int callCpuCCNF(){
	MatlabVec xVec, yVec, Precalc_Bs[3], Precalc_Bs_flat, Precalc_yBys, paramsVecs;
	matread("matfile_x.mat", "x", xVec); 
	matread("matfile_y.mat", "y", yVec);
	matread("matfile_Precalc_Bs_0.mat", "Precalc_Bs_1", Precalc_Bs[0]);
	matread("matfile_Precalc_Bs_1.mat", "Precalc_Bs_2", Precalc_Bs[1]);
	matread("matfile_Precalc_Bs_2.mat", "Precalc_Bs_3", Precalc_Bs[2]);
	matread("matfile_Precalc_Bs_flat.mat", "Precalc_Bs_flat_1", Precalc_Bs_flat);
	matread("matfile_Precalc_yBys.mat", "Precalc_yBys", Precalc_yBys);
	matread("matfile_params.mat", "params", paramsVecs);

	xEigen = Eigen::Map<Eigen::MatrixXd>(xVec.data.data(), xVec.M, xVec.N);
	yEigen = Eigen::Map<Eigen::MatrixXd>(yVec.data.data(), yVec.M, yVec.N);
	Precalc_Bs_0_Eigen = Eigen::Map<Eigen::MatrixXd>(Precalc_Bs[0].data.data(), Precalc_Bs[0].M, Precalc_Bs[0].N);
	Precalc_Bs_1_Eigen = Eigen::Map<Eigen::MatrixXd>(Precalc_Bs[1].data.data(), Precalc_Bs[1].M, Precalc_Bs[1].N);
	Precalc_Bs_2_Eigen = Eigen::Map<Eigen::MatrixXd>(Precalc_Bs[2].data.data(), Precalc_Bs[2].M, Precalc_Bs[2].N);
	Precalc_Bs_flatEigen = Eigen::Map<Eigen::MatrixXd>(Precalc_Bs_flat.data.data(), Precalc_Bs_flat.M, Precalc_Bs_flat.N);
	Precalc_yBysEigen = Eigen::Map<Eigen::MatrixXd>(Precalc_yBys.data.data(), Precalc_yBys.M, Precalc_yBys.N);

	VectorXd alpha = VectorXd::Constant(best_num_layer, 1.0);
	VectorXd betas = VectorXd::Constant(3, 1.0);
	MatrixXd thetas(best_num_layer, (input_layer_size + 1));
	std::vector<MatrixXd> thetas_good(num_reinit);
	std::vector<float> lhood(num_reinit);
	cublasCreate_v2(&cublasHd);
// 	for (int i = 0;  i < num_reinit; i++){
// 		srand((unsigned)time(NULL));
// 		thetas_good[i] = MatrixXd::Zero(best_num_layer, (input_layer_size + 1));
// 		randInitializeWeights(&thetas_good[i], input_layer_size, best_num_layer);		
// 		lhood[i] = LogLikelihoodCCNF(
// 			&yEigen,
// 			&xEigen,
// 			&alpha,
// 			&betas,
// 			&thetas_good[i],
// 			best_lambda_a,
// 			best_lambda_b,
// 			best_lambda_th,
// 			&Precalc_Bs_flatEigen,
// 			0,
// 			0,
// 			0,
// 			0,
// 			num_seqs,
// 			0
// 			);
// 		printf("%d %f\n", i, lhood[i]);
// 	}
	std::vector<float>::iterator result;
	result = std::max_element(lhood.begin(), lhood.end());
	int index = std::distance(lhood.begin(), result);
	thetas = thetas_good[index];
 
	VectorXd params = VectorXd::Zero(alpha.size() + betas.size() + thetas.rows() * thetas.cols());
	VectorXd B(Map<VectorXd>(thetas.data(), thetas.cols()*thetas.rows()));
//	params << alpha, betas, B;
	params = Eigen::Map<Eigen::VectorXd>(paramsVecs.data.data(), paramsVecs.M);

	const size_t NX = params.size();
	VectorXf params_f = params.cast <float>();
	cout << NX << endl;
	num_params = NX;
	// Use L-BFGS method to compute new sites
	const realreal epsg = EPSG;
	const realreal epsf = EPSF;
	const realreal epsx = EPSX;
	const int maxits = 200;
	stpscal = 2.75f; //Set for different problems!
	int info;

	//gpu buffer
	int* nbd;
	realreal* x;
	realreal* l;
	realreal* u;
	memAlloc<int>(&nbd, NX);
	memAlloc<realreal>(&x, NX);
	memAlloc<realreal>(&l, NX);
	memAlloc<realreal>(&u, NX);
	//cpu buffer
	int* nbd_ = (int*)malloc(sizeof(int)* NX);
	realreal* l_ = (realreal*)malloc(sizeof(realreal) * NX);
	realreal* u_ = (realreal*)malloc(sizeof(realreal) * NX);
	xx_opti = (realreal*)malloc(sizeof(realreal)* NX);

	for (int i = 0; i < NX; i++){
		nbd_[i] = 0;
	}
	cout << "alpha betas size" << (alpha.size() + betas.size()) << endl;
	for (int i = 0; i < (alpha.size() + betas.size()); i++){
		nbd_[i] = 1;
		l_[i] = 0.0;
	}
	memCopy(nbd, nbd_, NX * sizeof(int), cudaMemcpyHostToDevice);
	memCopy(l, l_, NX * sizeof(realreal), cudaMemcpyHostToDevice);
	memCopy(u, u_, NX * sizeof(realreal), cudaMemcpyHostToDevice);
	memCopy(x, params.data(), NX * sizeof(realreal), cudaMemcpyHostToDevice);

	printf("Start optimization...\n");
	stpscal = 1.0f;//2.75f;

	int	m = 8;
	if (NX < m)
		m = NX;
	lbfgsbminimize(NX, m, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
	printf("Ending code:%d\n", info);

	memFree(x);
	memFree(nbd);
	memFree(l);
	memFree(u);
	free(nbd_);
	free(l_);
	free(u_);
	free(xx_opti);
	cublasDestroy_v2(cublasHd);
	/*	// CPU
	cpu_ccnf rb1(NX);
	lbfgs minimizer(rb1);
	minimizer.setGradientEpsilon(1e-3f);
	minimizer.setMaxIterations(max_iter);
	lbfgs::status stat;
	
	stat = minimizer.minimize_with_host_x(params_f.data());
	cout << minimizer.statusToString(stat).c_str() << endl; */
	return 0;
}

void funcgrad(realreal* xxx, realreal& f, realreal* g, const cudaStream_t& stream){
	memCopy(xx_opti, xxx, sizeof(realreal)* num_params, cudaMemcpyDeviceToHost);

	static int count_ = 0;
	cout << "[" << count_ << "] begin" << endl;
	//VectorXf params_f = Map<Eigen::VectorXf>(xx_opti, m_numDimensions, 1);
	VectorXd params = Map<Eigen::VectorXd>(xx_opti, num_params, 1);
	double loss;
	VectorXd gradient = VectorXd::Zero(num_params);
	objectiveFunction(
		&loss,
		&gradient,
		&params,
		best_num_layer,
		3,
		best_num_layer,
		(input_layer_size + 1),
		best_lambda_a,
		best_lambda_b,
		best_lambda_th,
		&Precalc_Bs_0_Eigen,
		&Precalc_Bs_1_Eigen,
		&Precalc_Bs_2_Eigen,
		&xEigen,
		&yEigen,
		&Precalc_yBysEigen,
		&Precalc_Bs_flatEigen);
	f = (float)loss;

	cout << "[" << count_++ << "] " << f << endl;
// 	VectorXf params_ff = gradient.cast <float>();
// 	for (size_t i = 0; i < num_params; ++i){
// 		h_gradf[i] = params_ff[i];
// 	}

	memCopy(g, gradient.data(), sizeof(realreal)* num_params, cudaMemcpyHostToDevice);
	lbfgsbcuda::CheckBuffer(g, num_params, num_params);
}
int callGpuCCNF(){
	return 0;
}