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
	particle_pair[pd->id] = pd;
}

void xContactPairList::insertTriangleContactPair(xTrianglePairData* pd)
{
	triangle_pair[pd->id] = pd;
}

void xContactPairList::deletePlanePairData(unsigned int i)
{
	if (plane_pair.find(i) != plane_pair.end())
	{
		xPairData* pd = plane_pair.take(i);
		delete pd;
	}	
}

void xContactPairList::deleteParticlePairData(unsigned int i)
{
	if (particle_pair.find(i) != particle_pair.end())
	{
		xPairData* pd = particle_pair.take(i);
		delete pd;
	}	
}

void xContactPairList::deleteTrianglePairData(unsigned int i)
{
	if (triangle_pair.find(i) != triangle_pair.end())
	{
		xTrianglePairData* pd = triangle_pair.take(i);
		delete pd;
	}		
}

bool xContactPairList::IsNewPlaneContactPair(unsigned int i)
{
	return plane_pair.find(i) == plane_pair.end();
}

bool xContactPairList::IsNewParticleContactPair(unsigned int i)
{
	return particle_pair.find(i) == particle_pair.end();
}

bool xContactPairList::IsNewTriangleContactPair(unsigned int i)
{
	return triangle_pair.find(i) == triangle_pair.end();
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

QMap<unsigned int, xTrianglePairData*>& xContactPairList::TrianglePair()
{
	return triangle_pair;
}

xTrianglePairData* xContactPairList::TrianglePair(unsigned int i)
{
	return NULL;
}
