#include "xdynamics_algebra/xGridCell.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "thrust/sort.h"

vector3d xGridCell::wo;				// world origin
double xGridCell::cs = 0.0;			// cell size
vector3ui xGridCell::gs;				// grid size

xGridCell::xGridCell()
	: sorted_id(NULL)
	, cell_id(NULL)
	, body_id(NULL)
	, cell_start(NULL)
	, cell_end(NULL)
	, d_sorted_id(NULL)
	, d_cell_start(NULL)
	, d_cell_end(NULL)
	, d_cell_id(NULL)
	, d_body_id(NULL)
	, nse(0)
	, ng(0)
{

}

xGridCell::~xGridCell()
{
	clear();
}

void xGridCell::clear()
{
	if (cell_id) delete[] cell_id; cell_id = NULL;
	if (body_id) delete[] body_id; body_id = NULL;
	if (sorted_id) delete[] sorted_id; sorted_id = NULL;
	if (cell_start) delete[] cell_start; cell_start = NULL;
	if (cell_end) delete[] cell_end; cell_end = NULL;
	if (xSimulation::Gpu())
	{
		if (d_cell_id) checkCudaErrors(cudaFree(d_cell_id)); d_cell_id = NULL;
		if (d_body_id) checkCudaErrors(cudaFree(d_body_id)); d_body_id = NULL;
		if (d_sorted_id) checkCudaErrors(cudaFree(d_sorted_id)); d_sorted_id = NULL;
		if (d_cell_start) checkCudaErrors(cudaFree(d_cell_start)); d_cell_start = NULL;
		if (d_cell_end) checkCudaErrors(cudaFree(d_cell_end)); d_cell_start = NULL;
	}
}

void xGridCell::initialize(unsigned int np)
{
	allocMemory(np);
	if (xSimulation::Gpu())
		cuAllocMemory(np);
	else
	{
		d_cell_id = cell_id;
		d_body_id = body_id;
		d_sorted_id = sorted_id;
		d_cell_start = cell_start;
		d_cell_end = cell_end;
	}
}

void xGridCell::reorderDataAndFindCellStart(
	unsigned int id, unsigned int begin, unsigned int end)
{
	cell_start[id] = begin;
	cell_end[id] = end;
	unsigned dim = 0, bid = 0;
	for (unsigned i(begin); i < end; i++) {
		sorted_id[i] = body_id[i];
	}
}

void xGridCell::rearrange_cell()
{
	if(xSimulation::Gpu())
		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, nse,/* md->numPolygonSphere(),*/ ng);
	else
	{
		thrust::sort_by_key(cell_id, cell_id + nse, body_id);
		memset(cell_start, 0xffffffff, sizeof(unsigned int) * ng);
		memset(cell_end, 0, sizeof(unsigned int)*ng);
		unsigned int begin = 0, end = 0, id = 0;
		bool ispass;
		while (end++ != nse) {
			ispass = true;
			id = cell_id[begin];
			if (id != cell_id[end]) {
				end - begin > 1 ? ispass = false : reorderDataAndFindCellStart(id, begin++, end);
			}
			if (!ispass) {
				reorderDataAndFindCellStart(id, begin, end);
				begin = end;
			}
		}
	}
	
}

void xGridCell::allocMemory(unsigned int n)
{
	ng = gs.x * gs.y * gs.z;
	cell_id = new unsigned int[n]; memset(cell_id, 0, sizeof(unsigned int)*n);
	body_id = new unsigned int[n]; memset(body_id, 0, sizeof(unsigned int)*n);
	sorted_id = new unsigned int[n]; memset(sorted_id, 0, sizeof(unsigned int)*n);
	cell_start = new unsigned int[ng]; memset(cell_start, 0, sizeof(unsigned int)*ng);
	cell_end = new unsigned int[ng]; memset(cell_end, 0, sizeof(unsigned int)*ng);
	nse = n;
}

void xGridCell::cuAllocMemory(unsigned int n)
{
	ng = gs.x * gs.y * gs.z;
	checkCudaErrors(cudaMalloc((void**)&d_cell_id, sizeof(unsigned int)*n));	
	checkCudaErrors(cudaMalloc((void**)&d_body_id, sizeof(unsigned int)*n));
	checkCudaErrors(cudaMalloc((void**)&d_sorted_id, sizeof(unsigned int)*n));
	checkCudaErrors(cudaMalloc((void**)&d_cell_start, sizeof(unsigned int)*ng));
	checkCudaErrors(cudaMalloc((void**)&d_cell_end, sizeof(unsigned int)*ng));
	checkCudaErrors(cudaMemset(d_cell_id, 0, sizeof(unsigned int) * n));
	checkCudaErrors(cudaMemset(d_body_id, 0, sizeof(unsigned int) * n));
	checkCudaErrors(cudaMemset(d_sorted_id, 0, sizeof(unsigned int) * n));
	checkCudaErrors(cudaMemset(d_cell_start, 0, sizeof(unsigned int) * ng));
	checkCudaErrors(cudaMemset(d_cell_end, 0, sizeof(unsigned int) * ng));
	nse = n;
}

// void grid_base::cuResizeMemory(unsigned int n)
// {
// 	unsigned int *h_cell_id = new unsigned int[nse];
// 	unsigned int *h_body_id = new unsigned int[nse];
// 	unsigned int *h_sorted_id = new unsigned int[nse];
// 	checkCudaErrors(cudaFree(d_cell_id));
// 	checkCudaErrors(cudaFree(d_body_id));
// 	checkCudaErrors(cudaFree(d_sorted_id));
// 	checkCudaErrors(cudaMalloc((void**)&d_cell_id, sizeof(unsigned int) * n));
// 	checkCudaErrors(cudaMalloc((void**)&d_body_id, sizeof(unsigned int) * n));
// 	checkCudaErrors(cudaMalloc((void**)&d_sorted_id, sizeof(unsigned int) * n));
// 	checkCudaErrors(cudaMemcpy())
// }

// VEC3I grid_base::getCellNumber(double x, double y, double z)
// {
// 	return VEC3I(
// 		static_cast<int>(abs(std::floor((x - wo.x) / cs))),
// 		static_cast<int>(abs(std::floor((y - wo.y) / cs))),
// 		static_cast<int>(abs(std::floor((z - wo.z) / cs)))
// 		);
// }

vector3i xGridCell::getCellNumber(double x, double y, double z)
{
	return new_vector3i(
		static_cast<int>(abs(std::floor((x - wo.x) / cs))),
		static_cast<int>(abs(std::floor((y - wo.y) / cs))),
		static_cast<int>(abs(std::floor((z - wo.z) / cs)))
		);
}

unsigned int xGridCell::getHash(vector3i& c3)
{
	vector3i gridPos;
	gridPos.x = c3.x & (gs.x - 1);
	gridPos.y = c3.y & (gs.y - 1);
	gridPos.z = c3.z & (gs.z - 1);
	return (gridPos.z*gs.y) * gs.x + (gridPos.y*gs.x) + gridPos.x;
}
