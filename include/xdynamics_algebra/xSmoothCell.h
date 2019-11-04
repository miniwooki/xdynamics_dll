#ifndef XSMOOTHCELL_H
#define XSMOOTHCELL_H

#include "xdynamics_algebra/xGridCell.h"

class XDYNAMICS_API xSmoothCell : public xGridCell
{
public:
	xSmoothCell();
	~xSmoothCell();

	virtual void initialize(unsigned int np);
	virtual void detection(double *pos, unsigned int np, unsigned int sid);

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