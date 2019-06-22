#ifndef XROTATIONALAXIALFORCE_H
#define XROTATIONALAXIALFORCE_H

#include "xForce.h"

class xRotationalAxialForce : public xForce
{
public:
	xRotationalAxialForce();
	xRotationalAxialForce(std::string _name);
	virtual ~xRotationalAxialForce();

	void SetupDataFromStructure(xPointMass* ip, xPointMass* jp, xRotationalAxialForceData& d);

	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd);
	virtual void xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);
	virtual void xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);

private:
	double r_force;
	vector3d location;
	vector3d direction;
	vector3d f0_i, f0_j;
	vector3d f1_i, f1_j;
};

#endif