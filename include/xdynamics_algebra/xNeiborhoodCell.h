#ifndef XNEIBORHOODCELL_H
#define XNEIBORHOODCELL_H

#include "xdynamics_decl.h"
#include "xdynamics_algebra/xGridCell.h"

class XDYNAMICS_API xNeiborhoodCell : public xGridCell
{
public:
	xNeiborhoodCell();
	virtual ~xNeiborhoodCell();

	virtual void detection(double *pos = NULL, double* spos = NULL, unsigned int np = 0, unsigned int snp = 0);

private:
	void _detection(vector4d* pos, vector4d* spos, unsigned int np, unsigned int snp);
	void reorderDataAndFindCellStart(unsigned int id, unsigned int begin, unsigned int end);
};

#endif