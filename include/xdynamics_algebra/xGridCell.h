#ifndef XGRIDCELL_H
#define XGRIDCELL_H

#include "xdynamics_decl.h"

class XDYNAMICS_API xGridCell
{
public:
	xGridCell();
	virtual ~xGridCell();

	void clear();
	void initialize(unsigned int np);
	virtual void detection(double *pos = NULL, double* spos = NULL, unsigned int np = 0, unsigned int snp = 0) = 0;

	void setWorldOrigin(vector3d _wo) { wo = _wo; }
	void setCellSize(double _cs) { cs = _cs; }
	void setGridSize(vector3ui _gs) { gs = _gs; }
	unsigned int nCell() { return ng; }
	void allocMemory(unsigned int n);
	void cuAllocMemory(unsigned int n);
	static vector3i getCellNumber(double x, double y, double z);
	static unsigned int getHash(vector3i& c3);
	unsigned int sortedID(unsigned int id) { return sorted_id[id]; }
	unsigned int cellID(unsigned int id) { return cell_id[id]; }
	unsigned int bodyID(unsigned int id) { return body_id[id]; }
	unsigned int cellStart(unsigned int id) { return cell_start[id]; }
	unsigned int cellEnd(unsigned int id) { return cell_end[id]; }
	unsigned int* sortedID() { return d_sorted_id; }
	unsigned int* cellStart() { return d_cell_start; }
	unsigned int* cellEnd() { return d_cell_end; }

	static vector3d wo;			// world origin
	static double cs;			// cell size
	static vector3ui gs;			// grid size

protected:
	//Type type;
	unsigned int* sorted_id;
	unsigned int* cell_id;
	unsigned int* body_id;
	unsigned int* cell_start;
	unsigned int* cell_end;

	unsigned int *d_sorted_id;
	unsigned int *d_cell_id;
	unsigned int *d_body_id;
	unsigned int *d_cell_start;
	unsigned int *d_cell_end;

	unsigned int nse;   // number of sorting elements
	unsigned int ng;	// the number of grid
};

#endif