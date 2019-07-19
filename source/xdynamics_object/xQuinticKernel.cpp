#include "xdynamics_object/xQuinticKernel.h"

xQuinticKernel::xQuinticKernel()
	: xKernelFunction()
{
	
}

xQuinticKernel::~xQuinticKernel()
{

}

void xQuinticKernel::setupKernel(double gap, xKernelFunctionData& d)
{
	xKernelFunction::setupKernel(gap, d);
	kernel_support = 3;
	kernel_support_sq = kernel_support * kernel_support;
	if (d.dim == 3){
		kernel_const = 1.0 / (120.0 * M_PI) * h_inv_3;
	}
	else{
		kernel_const = 7.0 / (478.0 * M_PI * h_sq);
	}
	kernel_grad_const = (-5.0) * kernel_const * h_inv_sq;
}

double xQuinticKernel::sphKernel(double QSq)
{
	double Q = sqrt(QSq);
	if (Q < 1.0)
		return kernel_const * (pow(3.0 - Q, 5.0) - 6 * pow(2.0 - Q, 5.0) + 15 * pow(1.0 - Q, 5.0));
	else if (Q < 2.0)
		return kernel_const * (pow(3.0 - Q, 5.0) - 6 * pow(2.0 - Q, 5.0));
	else if (Q < 3.0)
		return kernel_const * pow(3.0 - Q, 5.0);

	return 0.0f;
}

vector3d xQuinticKernel::sphKernelGrad(double QSq, vector3d& distVec)
{
	double Q = sqrt(QSq);
	if (Q < 1.0)
		return (kernel_grad_const / Q) * (pow(3.0 - Q, 4.0) - 6 * pow(2.0 - Q, 4.0) + 15 * pow(1.00 - Q, 4.0)) * distVec;
	else if (Q < 2.0)
		return (kernel_grad_const / Q) * (pow(3.0 - Q, 4.0) - 6 * pow(2.0 - Q, 4.0)) * distVec;
	else if (Q < 3.0)
		return (kernel_grad_const / Q) * pow(3.0 - Q, 4.0) * distVec;

	return new_vector3d(0.0, 0.0, 0.0);
}