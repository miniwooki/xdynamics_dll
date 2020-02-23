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
	void ConvertGlobalToLocalOfBodyConnectionPosition(unsigned int i, xPointMass* pm);

	unsigned int NumSpringDamperConnection();
	unsigned int NumSpringDamperConnectionList();
	unsigned int NumSpringDamperConnectionValue();
	unsigned int NumSpringDamperBodyConnection();
	unsigned int NumSpringDamperBodyConnectionData();

	xSpringDamperConnectionInformation* xSpringDamperConnection();
	xSpringDamperConnectionData* xSpringDamperConnectionList();
	xSpringDamperCoefficient* xSpringDamperCoefficientValue();
	xSpringDamperBodyConnectionInfo* xSpringDamperBodyConnectionInformation();
	xSpringDamperBodyConnectionData* XSpringDamperBodyConnectionDataList();
	double* FreeLength();
	void initializeFreeLength(double* p,double* ep);

	static void xCalculateForce(
		vector3d spi, vector3d spj,
		double k, double c, double init_l,
		vector3d p0, vector3d p1,
		vector3d v0, vector3d v1,
		euler_parameters e0, euler_parameters e1,
		euler_parameters ev0, euler_parameters ev1,
		vector3d& f0, vector3d& f1,
		vector3d& m0, vector3d& m1
	);
	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd);
	void xCalculateForceForDEM(double* pos, double* vel, double* ep, double* ev, double* ms, double* force, double* moment);
	void xCalculateForceFromDEM(unsigned int ci, xPointMass* pm, const double* pos, const double* vel);
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
	unsigned int nBodyConnection;
	unsigned int nBodyConnectionData;
	xSpringDamperCoefficient *kc_value;
	xSpringDamperConnectionInformation* xsdci;
	xSpringDamperConnectionData *connection_data;
	xSpringDamperBodyConnectionInfo *connection_body_info;
	xSpringDamperBodyConnectionData *connection_body_data;
	double *free_length;
	/*double *dem_particle_position;
	double *dem_particle_velocity;*/
};

#endif