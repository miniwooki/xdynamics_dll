#ifndef XSPHERICALCONSTRAINT_H
#define XSPHERICALCONSTRAINT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xKinematicConstraint.h"

class XDYNAMICS_API xSphericalConstraint : public xKinematicConstraint
{
public:
	xSphericalConstraint();
	xSphericalConstraint(std::string _name, std::string _i, std::string _j);
	virtual ~xSphericalConstraint();

	virtual void ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul);
	virtual void ConstraintJacobian(xSparseD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr);
	virtual void DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul);
	virtual kinematicConstraint_result GetStepResult(unsigned int part, xVectorD& q, xVectorD& qd, double* L, unsigned int sr);
	virtual void GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul);
};

#endif