#include "xdynamics_object/xQuadraticKernel.h"

xQuadraticKernel::xQuadraticKernel()
	: xKernelFunction()
{
	
}

xQuadraticKernel::~xQuadraticKernel()
{

}

void xQuadraticKernel::setupKernel(double gap, xKernelFunctionData& d)
{
	xKernelFunction::setupKernel(gap, d);
	kernel_support = 2.5;
	kernel_support_sq = kernel_support * kernel_support;
	if (d.dim == 3){
		kernel_const = (1.0 / (120.0 * M_PI)) * h_inv_3;
	}
	else{
		kernel_const = 96.0 / (1199.0 * M_PI * h_sq);
	}
	kernel_grad_const = (-4.0) * kernel_const * h_inv_sq;
}

double xQuadraticKernel::sphKernel(double QSq)
{
	double Q = sqrt(QSq);
	if (Q < 0.5)
		return kernel_const * (pow(2.5 - Q, 4.0) - 5 * pow(1.50 - Q, 4.0) + 10 * pow(0.50 - Q, 4.0));
	else if (Q < 1.5)
		return kernel_const * (pow(2.5 - Q, 4.0) - 5 * pow(1.5 - Q, 4.0));
	else if (Q < 2.5)
		return kernel_const * pow(2.5 - Q, 4.0);

	return 0.0;
}

vector3d xQuadraticKernel::sphKernelGrad(double QSq, vector3d& distVec)
{
	double Q = sqrt(QSq);
	if (Q < 0.5)
		return (kernel_grad_const / Q) * (pow(2.5 - Q, 3.0) - 5 * pow(1.50 - Q, 3.0) + 10 * pow(0.50 - Q, 3.0)) * distVec;
	else if (Q < 1.5)
		return (kernel_grad_const / Q) * (pow(2.5 - Q, 3.0) - 5 * pow(1.50 - Q, 3.0)) * distVec;
	else if (Q < 2.5)
		return (kernel_grad_const / Q) * pow(2.5 - Q, 3.0) * distVec;

	return new_vector3d(0.0, 0.0, 0.0);
}