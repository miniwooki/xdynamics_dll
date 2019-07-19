#include "xdynamics_algebra/xSmoothCell.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include "xdynamics_parallel/xParallelSPH_decl.cuh"

xSmoothCell::xSmoothCell()
	: xGridCell()
{

}

xSmoothCell::~xSmoothCell()
{

}

void xSmoothCell::setWorldBoundary(vector3d bMin, vector3d bMax)
{
	min_grid = bMin;
	max_grid = bMax;
}

vector3d xSmoothCell::MinimumGridPosition()
{
	return min_grid;
}

vector3d xSmoothCell::MaximumGridPosition()
{
	return max_grid;
}

vector3d xSmoothCell::GridSize()
{
	return grid_size;
}

void xSmoothCell::initialize(unsigned int np)
{
	min_grid = min_grid - new_vector3d(cs, cs, cs);
	max_grid = max_grid + new_vector3d(cs, cs, cs);
	grid_size = max_grid - min_grid;

	gs.x = static_cast<unsigned int>(ceil(grid_size.x / cs));
	gs.y = static_cast<unsigned int>(ceil(grid_size.y / cs));
	gs.z = static_cast<unsigned int>(ceil(grid_size.z / cs));
	
	xGridCell::initialize(np);


}

void xSmoothCell::detection(double *pos /* = NULL */, double* spos /* = NULL */, unsigned int np /* = 0 */, unsigned int snp /* = 0 */)
{

}

