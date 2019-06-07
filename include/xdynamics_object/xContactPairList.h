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
	void deletePlanePairData(unsigned int i);
	void deleteParticlePairData(unsigned int i);
	void deleteTrianglePairData(unsigned int i);

	bool IsNewPlaneContactPair(unsigned int i);
	bool IsNewParticleContactPair(unsigned int i);
	bool IsNewTriangleContactPair(unsigned int i);

	QMap<unsigned int, xPairData*>& PlanePair();
	QMap<unsigned int, xPairData*>& ParticlePair();
	QMap<unsigned int, xTrianglePairData*>& TrianglePair();

	xPairData* PlanePair(unsigned int i);
	xPairData* ParticlePair(unsigned int i);
	xTrianglePairData* TrianglePair(unsigned int i);

private:
	QMap<unsigned int, xPairData*> plane_pair;
	QMap<unsigned int, xPairData*> particle_pair;
	QMap<unsigned int, xTrianglePairData*> triangle_pair;
};

#endif