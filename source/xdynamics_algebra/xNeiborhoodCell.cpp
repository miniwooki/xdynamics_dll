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
	//	unsigned int _np = 0;
	// 	if (md->particleSystem()->particleCluster().size())
	// 		_np = md->particleSystem()->particleCluster().size() * particle_cluster::perCluster();
	// 	else
	// 		_np = md->numParticle();
	for (unsigned int i = 0; i < np; i++){
		unsigned int idx = sid + i;
		vector4d p = pos[i];
		cell3d = getCellNumber(p.x, p.y, p.z);
		cell_id[idx] = getHash(cell3d);
		body_id[idx] = sid + i;
	}
	/*if (spos)
	{
		for (unsigned int i = 0; i < snp; i++){
			cell3d = getCellNumber(spos[i].x, spos[i].y, spos[i].z);
			cell_id[np + i] = getHash(cell3d);
			body_id[np + i] = np + i;
		}
	}*/
	
}

void xNeiborhoodCell::detection(double *pos, unsigned int np, unsigned int sid)
{
	if (xSimulation::Gpu())
	{
		cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, np);
		//	qDebug() << "detection0 done";
		//if (snp && spos)
		//{
		//	cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, np, snp, spos);
		//	// 		//	qDebug() << "detection1 done";
		//}

		
	}
	else
		_detection((vector4d*)pos, np, sid);
}

// void xNeiborhoodCell::detection_f(float *pos /*= NULL*/, float* spos /*= NULL*/, unsigned int np /*= 0*/, unsigned int snp /*= 0*/)
//{
//	if (simulation::isGpu())
//	{
//		cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, np);
//		if (snp && spos)
//			cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, np, snp, spos);
//		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, np + snp,/* md->numPolygonSphere(),*/ ng);
//	}
//	else
//		_detection_f((VEC4F_PTR)pos, (VEC4F_PTR)spos, np, snp);
//}