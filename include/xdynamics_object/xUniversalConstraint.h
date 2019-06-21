#ifndef XUNIVERSALCONSTRAINT_H
#define XUNIVERSALCONSTRAINT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xKinematicConstraint.h"

class XDYNAMICS_API xUniversalConstraint : public xKinematicConstraint
{
public:
	xUniversalConstraint();
	xUniversalConstraint(std::wstring _name, std::wstring _i, std::wstring _j);
	virtual ~xUniversalConstraint();

	virtual void ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul);
	virtual void ConstraintJacobian(xSparseD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr);
	virtual void DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul);
	virtual void SaveStepResult(unsigned int part, double ct, xVectorD& q, xVectorD& qd, double* L, unsigned int sr);
	virtual void GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul);
};

#endif