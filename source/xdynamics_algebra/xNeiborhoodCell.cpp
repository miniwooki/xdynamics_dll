#include "xdynamics_algebra/xNeiborhoodCell.h"
#include "thrust/sort.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"

xNeiborhoodCell::xNeiborhoodCell()
	: xGridCell()
{

}

xNeiborhoodCell::~xNeiborhoodCell()
{
	
}

void xNeiborhoodCell::reorderDataAndFindCellStart(unsigned int id, unsigned int begin, unsigned int end)
{
	cell_start[id] = begin;
	cell_end[id] = end;
	unsigned dim = 0, bid = 0;
	for (unsigned i(begin); i < end; i++){
		sorted_id[i] = body_id[i];
	}
}

void xNeiborhoodCell::_detection(vector4d* pos, vector4d* spos, unsigned int np, unsigned int snp)
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
		//		unsigned int rid = rearranged_id[i];
		vector4d p = pos[i];
		cell3d = getCellNumber(p.x, p.y, p.z);
		cell_id[i] = getHash(cell3d);
		body_id[i] = i;
	}
	if (spos)
	{
		for (unsigned int i = 0; i < snp; i++){
			cell3d = getCellNumber(spos[i].x, spos[i].y, spos[i].z);
			cell_id[np + i] = getHash(cell3d);
			body_id[np + i] = np + i;
		}
	}
	thrust::sort_by_key(cell_id, cell_id + np + snp, body_id);
	memset(cell_start, 0xffffffff, sizeof(unsigned int) * ng);
	memset(cell_end, 0, sizeof(unsigned int)*ng);
	unsigned int begin = 0, end = 0, id = 0;
	bool ispass;
	while (end++ != np + snp){
		ispass = true;
		id = cell_id[begin];
		if (id != cell_id[end]){
			end - begin > 1 ? ispass = false : reorderDataAndFindCellStart(id, begin++, end);
		}
		if (!ispass){
			reorderDataAndFindCellStart(id, begin, end);
			begin = end;
		}
	}
}

void xNeiborhoodCell::detection(double *pos, double* spos, unsigned int np, unsigned int snp)
{
	if (xSimulation::Gpu())
	{
		cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, np);
		//	qDebug() << "detection0 done";
		if (snp && spos)
		{
			cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, np, snp, spos);
			// 		//	qDebug() << "detection1 done";
		}
		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, np + snp,/* md->numPolygonSphere(),*/ ng);
	}
	else
		_detection((vector4d*)pos, (vector4d*)spos, np, snp);
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