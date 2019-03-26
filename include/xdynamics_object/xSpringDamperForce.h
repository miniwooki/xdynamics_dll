#ifndef XSPRINGDAMPERFORCE_H
#define XSPRINGDAMPERFORCE_H

#include "xdynamics_object/xForce.h"

class xSpringDamperForce : public xForce
{
public:
	xSpringDamperForce();
	xSpringDamperForce(std::string _name);
	virtual ~xSpringDamperForce();

	void SetupDataFromStructure(xTSDAData& d);

	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd, xVectorD& rhs);
	virtual void xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);
	virtual void xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);

private:
	double init_l;
	double k;
	double c;

	vector3d L;
	double f;
	double l;
	double dl;
};

#endif