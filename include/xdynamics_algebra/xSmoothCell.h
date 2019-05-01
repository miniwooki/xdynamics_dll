#ifndef XSMOOTHCELL_H
#define XSMOOTHCELL_H

#include "xdynamics_algebra/xGridCell.h"

class XDYNAMICS_API xSmoothCell : public xGridCell
{
public:
	xSmoothCell();
	~xSmoothCell();

	virtual void initialize(unsigned int np);
	virtual void detection(double *pos = NULL, double* spos = NULL, unsigned int np = 0, unsigned int snp = 0);

	void setWorldBoundary(vector3d bMin, vector3d bMax);
	vector3d MinimumGridPosition();
	vector3d MaximumGridPosition();
	vector3d GridSize();

private:
	vector3d min_grid;
	vector3d max_grid;
	vector3d grid_size;

};

#endif