#include "xdynamics_object/xKernelFunction.h"

xKernelFunction::xKernelFunction()
{
// 	h = sph->KernelLength();
// 	h_sq = h * h;
// 	h_inv = 1.0 / h;
// 	h_inv_sq = 1.0 / (h * h);
// 	h_inv_2 = 1.0 / h / h;
// 	h_inv_3 = 1.0 / pow(h, 3);
// 	h_inv_4 = 1.0 / pow(h, 4);
// 	h_inv_5 = 1.0 / pow(h, 5);
}

xKernelFunction::~xKernelFunction()
{

}

void xKernelFunction::setupKernel(double gap, xKernelFunctionData& d)
{
	h = gap * d.factor;
	h_sq = h * h;
	h_inv = 1.0 / h;
	h_inv_sq = 1.0 / (h * h);
	h_inv_2 = 1.0 / h / h;
	h_inv_3 = 1.0 / pow(h, 3);
	h_inv_4 = 1.0 / pow(h, 4);
	h_inv_5 = 1.0 / pow(h, 5);
}

double xKernelFunction::KernelConst()
{
	return kernel_const;
}

double xKernelFunction::KernelGradConst()
{
	return kernel_grad_const;
}

double xKernelFunction::KernelSupport()
{
	return kernel_support;
}

double xKernelFunction::KernelSupprotSq()
{
	return kernel_support_sq;
}
