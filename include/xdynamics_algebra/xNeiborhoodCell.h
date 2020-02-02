#ifndef XNEIBORHOODCELL_H
#define XNEIBORHOODCELL_H

/*#include "xdynamics_decl.h"*/
#include "xdynamics_algebra/xGridCell.h"

class XDYNAMICS_API xNeiborhoodCell : public xGridCell
{
public:
	xNeiborhoodCell();
	virtual ~xNeiborhoodCell();

	virtual void detection(double *pos, unsigned int np, unsigned int sid);
	void detectionCpu(double* pos, unsigned int np, unsigned int sid);

private:
	void _detection(vector4d* pos, unsigned int np, unsigned int sid);
	
};

#endif