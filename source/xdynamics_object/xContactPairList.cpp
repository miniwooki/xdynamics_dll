#include "xdynamics_object/xContactPairList.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include <QtCore/QList>

xContactPairList::xContactPairList()
{

}

xContactPairList::~xContactPairList()
{
	qDeleteAll(plane_pair);
	qDeleteAll(particle_pair);
	qDeleteAll(cylinder_pair);
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

void xContactPairList::insertCylinderContactPair(xPairData * pd)
{
	cylinder_pair[pd->id] = pd;
}

void xContactPairList::insertParticleContactPair(xPairData* pd)
{
	particle_pair[pd->id] = pd;
}

void xContactPairList::insertTriangleContactPair(xTrianglePairData* pd)
{
	triangle_pair[pd->id] = pd;
}

void xContactPairList::insertTriangleLineContactPair(xTrianglePairData * pd)
{
	triangle_line_pair[pd->id] = pd;
}

void xContactPairList::insertTrianglePointContactPair(xTrianglePairData * pd)
{
	triangle_point_pair[pd->id] = pd;
}

void xContactPairList::deletePlanePairData(unsigned int i)
{
	if (plane_pair.find(i) != plane_pair.end())
	{
		xPairData* pd = plane_pair.take(i);
		delete pd;
	}	
}

void xContactPairList::deleteCylinderPairData(unsigned int i)
{
	if (cylinder_pair.find(i) != cylinder_pair.end())
	{
		xPairData* pd = cylinder_pair.take(i);
		delete pd;
	}
}

void xContactPairList::deleteParticlePairData(unsigned int i)
{
	if (particle_pair.size())
		i = i;
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

void xContactPairList::deleteTriangleLinePairData(unsigned int i)
{
	if (triangle_line_pair.find(i) != triangle_line_pair.end())
	{
		xTrianglePairData* pd = triangle_line_pair.take(i);
		delete pd;
	}
}

void xContactPairList::deleteTrianglePointPairData(unsigned int i)
{
	if (triangle_point_pair.find(i) != triangle_point_pair.end())
	{
		xTrianglePairData* pd = triangle_point_pair.take(i);
		delete pd;
	}
}

bool xContactPairList::IsNewPlaneContactPair(unsigned int i)
{
	return plane_pair.find(i) == plane_pair.end();
}

bool xContactPairList::IsNewCylinderContactPair(unsigned int i)
{
	return cylinder_pair.find(i) == cylinder_pair.end();
}

bool xContactPairList::IsNewParticleContactPair(unsigned int i)
{
	return particle_pair.find(i) == particle_pair.end();
}

bool xContactPairList::IsNewTriangleContactPair(unsigned int i)
{
	return triangle_pair.find(i) == triangle_pair.end();
}

bool xContactPairList::IsNewTriangleLineContactPair(unsigned int i)
{
	return triangle_line_pair.find(i) == triangle_line_pair.end();
}

bool xContactPairList::IsNewTrianglePointContactPair(unsigned int i)
{
	return triangle_point_pair.find(i) == triangle_point_pair.end();
}

bool xContactPairList::TriangleContactCheck(double r, vector3d& pos, xTrianglePairData * d)
{
	vector3d rp = new_vector3d(d->cpx, d->cpy, d->cpz) - pos;
	double dist = length(rp);
	bool b = r - dist;
	return b;
}

QMap<unsigned int, xPairData*>& xContactPairList::PlanePair()
{
	return plane_pair;
}

QMap<unsigned int, xPairData*>& xContactPairList::CylinderPair()
{
	return cylinder_pair;
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
	QMap<unsigned int, xPairData*>::const_iterator it = particle_pair.find(i);
	if (it == particle_pair.constEnd())
		return NULL;
	return particle_pair[i];
}

xPairData * xContactPairList::CylinderPair(unsigned int i)
{
	QMap<unsigned int, xPairData*>::const_iterator it = cylinder_pair.find(i);
	if (it == cylinder_pair.constEnd())
		return NULL;
	return cylinder_pair[i];
}

QMap<unsigned int, xTrianglePairData*>& xContactPairList::TrianglePair()
{
	return triangle_pair;
}

QMap<unsigned int, xTrianglePairData*>& xContactPairList::TriangleLinePair()
{
	return triangle_line_pair;
}

QMap<unsigned int, xTrianglePairData*>& xContactPairList::TrianglePointPair()
{
	return triangle_point_pair;
}

xTrianglePairData* xContactPairList::TrianglePair(unsigned int i)
{
	QMap<unsigned int, xTrianglePairData*>::const_iterator it = triangle_pair.find(i);
	if (it == triangle_pair.constEnd())
		return NULL;

	return triangle_pair[i];
}

xTrianglePairData * xContactPairList::TriangleLinePair(unsigned int i)
{
	QMap<unsigned int, xTrianglePairData*>::const_iterator it = triangle_line_pair.find(i);
	if (it == triangle_line_pair.constEnd())
		return NULL;

	return triangle_line_pair[i];
}

xTrianglePairData * xContactPairList::TrianglePointPair(unsigned int i)
{
	QMap<unsigned int, xTrianglePairData*>::const_iterator it = triangle_point_pair.find(i);
	if (it == triangle_point_pair.constEnd())
		return NULL;

	return triangle_point_pair[i];
}
