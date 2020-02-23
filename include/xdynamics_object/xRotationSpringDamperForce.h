#ifndef XROTATIONSPRINGDAMPERFORCE_H
#define XROTATIONSPRINGDAMPERFORCE_H

#include "xdynamics_object/xForce.h"

class xRotationSpringDamperForce : public xForce
{
public:
	xRotationSpringDamperForce();
	xRotationSpringDamperForce(std::string _name);
	virtual ~xRotationSpringDamperForce();

	void SetupDataFromStructure(xPointMass* ip, xPointMass* jp, xRSDAData& d);
	void SetupDataFromListData(xRSDAData&d, std::string data);
	void ConvertGlobalToLocalOfBodyConnectionPosition(unsigned int i, xPointMass* pm);

	size_t NumSpringDamperConnection();
	size_t NumSpringDamperConnectionList();
	size_t NumSpringDamperConnectionValue();
	size_t NumSpringDamperBodyConnection();
	size_t NumSpringDamperBodyConnectionData();

	xRSDAConnectionInformation* xSpringDamperConnection();
	xRSDAConnectionData* xSpringDamperConnectionList();
	xSpringDamperCoefficient* xSpringDamperCoefficientValue();
	xSpringDamperBodyConnectionInfo* xSpringDamperBodyConnectionInformation();
	xRSDABodyAttachedData* XSpringDamperBodyConnectionDataList();
	double* FreeAngle();
	void initializeAttachedPointForDEM(double* p, double* ep);

	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd);
	void xCalculateForceForDEM(double* pos, double* vel, double* ep, double* ev, double* ms, double* force, double* moment);
	void xCalculateForceFromDEM(unsigned int ci, xPointMass* pm, const double* pos, const double* vel);
	virtual void xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);
	virtual void xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul);

private:
	double init_theta;
	double k;
	double c;
	vector3d f_i;
	vector3d f_j;
	vector3d g_i;
	vector3d g_j;
	vector3d h_i;
	vector3d h_j;
	vector3d loc;

	vector3d L;
	double f;
	double l;
	double dl;

	size_t nrdci;
	size_t nkcvalue;
	size_t nConnection;
	size_t nBodyAttached;
	size_t nBodyAttachedData;
	xSpringDamperCoefficient *kc_value;
	xRSDAConnectionInformation* xrdci;
	xRSDAConnectionData *connection_data;
	xSpringDamperBodyConnectionInfo *attached_body_info;
	xRSDABodyAttachedData *attached_body_data;
	double *free_angle;

	int udrl;
	int n_rev;
	double theta;
	double dtheta;
	double n;
	/*double *dem_particle_position;
	double *dem_particle_velocity;*/
};

#endif