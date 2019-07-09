#ifndef XCONTACTPAIRLIST_H
#define XCONTACTPAIRLIST_H

#include "xdynamics_decl.h"
#include <QtCore/QMap>

class XDYNAMICS_API xContactPairList
{
public:
	xContactPairList();
	~xContactPairList();

	void insertPlaneContactPair(xPairData* pd);
	void insertParticleContactPair(xPairData* pd);
	void insertTriangleContactPair(xTrianglePairData* pd);
	void insertTriangleLineContactPair(xTrianglePairData* pd);
	void insertTrianglePointContactPair(xTrianglePairData* pd);

	void deletePlanePairData(unsigned int i);
	void deleteParticlePairData(unsigned int i);
	void deleteTrianglePairData(unsigned int i);
	void deleteTriangleLinePairData(unsigned int i);
	void deleteTrianglePointPairData(unsigned int i);

	bool IsNewPlaneContactPair(unsigned int i);
	bool IsNewParticleContactPair(unsigned int i);
	bool IsNewTriangleContactPair(unsigned int i);
	bool IsNewTriangleLineContactPair(unsigned int i);
	bool IsNewTrianglePointContactPair(unsigned int i);

	bool TriangleContactCheck(double r, vector3d& pos, xTrianglePairData* d);

	QMap<unsigned int, xPairData*>& PlanePair();
	QMap<unsigned int, xPairData*>& ParticlePair();
	QMap<unsigned int, xTrianglePairData*>& TrianglePair();
	QMap<unsigned int, xTrianglePairData*>& TriangleLinePair();
	QMap<unsigned int, xTrianglePairData*>& TrianglePointPair();

	xPairData* PlanePair(unsigned int i);
	xPairData* ParticlePair(unsigned int i);
	xTrianglePairData* TrianglePair(unsigned int i);
	xTrianglePairData* TriangleLinePair(unsigned int i);
	xTrianglePairData* TrianglePointPair(unsigned int i);

private:
	QMap<unsigned int, xPairData*> plane_pair;
	QMap<unsigned int, xPairData*> particle_pair;
	QMap<unsigned int, xTrianglePairData*> triangle_pair;
	QMap<unsigned int, xTrianglePairData*> triangle_line_pair;
	QMap<unsigned int, xTrianglePairData*> triangle_point_pair;
};

#endif