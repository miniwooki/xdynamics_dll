#ifndef XFORCE_H
#define XFORCE_H

#include "xdynamics_decl.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_object/xPointMass.h"

class xForce
{
public:
	enum fType{ NO_TYPE = -1, TSDA = 0, RSDA = 1, RAXIAL = 2, TSDA_LIST_DATA = 99 };
	xForce();
	xForce(std::string _name, fType _type);
	virtual ~xForce();

	void setBaseBodyName(std::string bn);
	void setActionBodyName(std::string an);
	void setBaseBodyIndex(int _i);
	void setActionBodyIndex(int _j);
// 	void setBaseLocalCoordinate(vector3d _spi);
// 	void setActionLocalCoordinate(vector3d _spj);

	std::string Name();
	fType Type();
	std::string BaseBodyName();
	std::string ActionBodyName();

	virtual void xCalculateForce(const xVectorD& q, const xVectorD& qd) = 0;
	//virtual vector3d xCalculateForceForDEM(vector3d& ip, vector3d& jp, vector3d& iv, vector3d& jv) = 0;
	virtual void xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul) = 0;
	virtual void xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul) = 0;

protected:
	fType type;
	xstring name;
	xstring base, action;
	int i, j;
	xPointMass *i_ptr;
	xPointMass *j_ptr;
	vector3d spi, spj;
};

#endif