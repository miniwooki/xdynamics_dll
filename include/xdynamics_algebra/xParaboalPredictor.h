#ifndef XPARABOLAPREDICTOR_H
#define XPARABOLAPREDICTOR_H

#include "xdynamics_decl.h"

class XDYNAMICS_API xParabolaPredictor
{
public:
	xParabolaPredictor();
	~xParabolaPredictor();

	bool apply(unsigned int it);

	double& getTimeStep() { return dt; }

	void init(double* _data, int _dataSize);

private:
	vector3i idx;
	vector3d xp;
	vector3d yp;
	vector3d coeff;
	matrix33d A;

	xVectorD* data3;

	int dataSize;

	double* data;
	double dt;
};

#endif