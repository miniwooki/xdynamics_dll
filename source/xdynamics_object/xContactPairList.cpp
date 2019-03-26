#include "xdynamics_object/xContactPairList.h"
#include <QtCore/QList>

xContactPairList::xContactPairList()
{

}

xContactPairList::~xContactPairList()
{
	qDeleteAll(plane_pair);
	qDeleteAll(particle_pair);
}

void xContactPairList::insertPlaneContactPair(xPairData* pd)
{
	if (plane_pair[pd->id])
		plane_pair[pd->id]->count += 1;
	else
		plane_pair[pd->id] = pd;
// 	QList<unsigned int> keys = plane_pair.keys();//QStringList keys = objects.keys();
// 	QList<unsigned int>::const_iterator it = qFind(plane_pair.keys(), pd.id);
// 	if (it == keys.end() || !keys.size())
// 		return NULL;
// 	plane_pair[pd.id] = pd;
}

void xContactPairList::insertParticleContactPair(xPairData* pd)
{
	//particle_pair[pd.id] = pd;
}

void xContactPairList::deletePlanePairData(unsigned int i)
{
	xPairData* pd = plane_pair.take(i);
	delete pd;
}

void xContactPairList::deleteParticlePairData(unsigned int i)
{
	xPairData* pd = particle_pair.take(i);
	delete pd;
}

QMap<unsigned int, xPairData*>& xContactPairList::PlanePair()
{
	return plane_pair;
}

xPairData* xContactPairList::PlanePair(unsigned int i)
{
	QMap<unsigned int, xPairData*>::const_iterator it = plane_pair.find(i);
	if (it == plane_pair.constEnd())
		return NULL;
	return plane_pair[i];
}

QMap<unsigned int, xPairData*>& xContactPairList::ParticlePair()
{
	return particle_pair;
}

xPairData* xContactPairList::ParticlePair(unsigned int i)
{
	return NULL;
}
