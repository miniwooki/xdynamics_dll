#include "xdynamics_object/xWendlandKernel.h"


xWendlandKernel::xWendlandKernel()
	: xKernelFunction()
{
	
}

xWendlandKernel::~xWendlandKernel()
{

}

void xWendlandKernel::setupKernel(double gap, xKernelFunctionData& d)
{
	xKernelFunction::setupKernel(gap, d);
	kernel_support = 2.0;
	kernel_support_sq = kernel_support * kernel_support;
	if (d.dim == 3){
		kernel_const = 21.0 / (16.0 * M_PI) * h_inv_3;
	}
	else{
		kernel_const = 7.0 / (4.0 * M_PI * h_sq);
	}
	kernel_grad_const = (-5.0) * kernel_const * h_inv_sq;
}

double xWendlandKernel::sphKernel(double QSq)
{
	double Q = sqrt(QSq);
	if (Q <= 2.0)
		return kernel_const * pow((1 - 0.5 * Q), 4)*(1 + 2.0 * Q);

	return 0.0;
}

vector3d xWendlandKernel::sphKernelGrad(double QSq, vector3d& distVec)
{
	double Q = sqrt(QSq);
	if (Q <= 2.0)
		return (kernel_grad_const / Q) * Q * pow((1 - 0.5 * Q), 3) * distVec;

	return new_vector3d(0.0,0.0,0.0);
}