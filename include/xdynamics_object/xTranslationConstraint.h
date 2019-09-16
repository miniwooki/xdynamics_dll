#ifndef XTRANSLATIONCONSTRAINT_H
#define XTRANSLATIONCONSTRAINT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xKinematicConstraint.h"

class XDYNAMICS_API xTranslationConstraint : public xKinematicConstraint
{
public:
	xTranslationConstraint();
	xTranslationConstraint(std::string _name, std::string _i, std::string _j);
	virtual ~xTranslationConstraint();

	virtual void ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul);
	virtual void ConstraintJacobian(xSparseD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr);
	virtual void DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul);
	virtual kinematicConstraint_result GetStepResult(unsigned int part, xVectorD& q, xVectorD& qd, double* L, unsigned int sr);
	virtual void GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul);
};

#endif