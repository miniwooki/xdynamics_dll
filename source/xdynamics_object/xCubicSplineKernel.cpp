#include "xdynamics_object/xCubicSplineKernel.h"

xCubicSplineKernel::xCubicSplineKernel()
	: xKernelFunction()
{
	
}

xCubicSplineKernel::~xCubicSplineKernel()
{

}

void xCubicSplineKernel::setupKernel(double gap, xKernelFunctionData& d)
{
	xKernelFunction::setupKernel(gap, d);
	kernel_support = 2.0;
	kernel_support_sq = kernel_support * kernel_support;
	if (d.dim = 3){
		kernel_const = 1 / (M_PI)* h_inv_3;
	}
	else{
		kernel_const = 10.0 / (7.0 * M_PI * h_sq);
	}
	kernel_grad_const = (-3.0 / 4.0) * kernel_const * h_inv_sq;
}

double xCubicSplineKernel::sphKernel(double QSq)
{
	double Q = sqrt(QSq);
	if (0 <= Q  && Q <= 1.0)
		return kernel_const * (1.0 - 1.5 * QSq + 0.75 * QSq * Q);
	else if (1.0 <= Q && Q <= 2.0)
		return kernel_const * 0.25 * pow(2.0 - Q, 3.0);

	return 0.0;
}

vector3d xCubicSplineKernel::sphKernelGrad(double QSq, vector3d& distVec)
{
	double Q = sqrt(QSq);
	vector3d grad;
	if (Q <= 1.0)
		return kernel_grad_const * (4.0 - 3.0 * Q) * distVec;
	else {
		double dif = 2.0 - Q;
		return kernel_grad_const * dif * dif * (distVec / Q);
	}
	return new_vector3d(0.0, 0.0, 0.0);
}