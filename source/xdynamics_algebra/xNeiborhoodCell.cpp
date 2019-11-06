#include "xdynamics_algebra/xNeiborhoodCell.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"

xNeiborhoodCell::xNeiborhoodCell()
	: xGridCell()
{

}

xNeiborhoodCell::~xNeiborhoodCell()
{
	
}

void xNeiborhoodCell::_detection(vector4d* pos, unsigned int np, unsigned int sid)
{
	//VEC4D_PTR pos = md->particleSystem()->position();
	vector4d *psph = NULL;
	vector3i cell3d;
	for (unsigned int i = 0; i < np; i++){
		unsigned int idx = sid + i;
		vector4d p = pos[i];
		cell3d = getCellNumber(p.x, p.y, p.z);
		cell_id[idx] = getHash(cell3d);
		body_id[idx] = sid + i;
	}
}

void xNeiborhoodCell::detection(double *pos, unsigned int np, unsigned int sid)
{
	if (xSimulation::Gpu())
	{
		cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, sid, np);
	}
	else
		_detection((vector4d*)pos, np, sid);
}