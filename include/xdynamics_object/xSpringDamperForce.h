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
	void SetupDataFromListData(xTSDAData&d, std::string data);

	unsigned int NumSpringDamperConnection();
	unsigned int NumSpringDamperConnectionList();
	unsigned int NumSpringDamperConnectionValue();

	xSpringDamperConnectionInformation* xSpringDamperConnection();
	xSpringDamperConnectionData* xSpringDamperConnectionList();
	xSpringDamperCoefficient* xSpringDamperCoefficientValue();
	double* FreeLength();
	void initializeFreeLength(double* p);

	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd);
	void xCalculateForceForDEM(double* pos, double* vel, double* force);
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

	unsigned int nsdci;
	unsigned int nkcvalue;
	unsigned int nConnection;
	xSpringDamperCoefficient *kc_value;
	xSpringDamperConnectionInformation* xsdci;
	xSpringDamperConnectionData *connection_data;
	double *free_length;
};

#endif