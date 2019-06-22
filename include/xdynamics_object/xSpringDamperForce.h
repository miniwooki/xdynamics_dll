#ifndef XSPRINGDAMPERFORCE_H
#define XSPRINGDAMPERFORCE_H

#include "xdynamics_object/xForce.h"

class xSpringDamperForce : public xForce
{
public:
	xSpringDamperForce();
	xSpringDamperForce(std::string _name);
	virtual ~xSpringDamperForce();

	void SetupDataFromStructure(xPointMass* ip, xPointMass* jp, xTSDAData& d);

	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd);
	virtual void xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);
	virtual void xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);

private:
	double init_l;
	double k;
	double c;
	vector3d loc_i;
	vector3d loc_j;

	vector3d L;
	double f;
	double l;
	double dl;
};

#endif