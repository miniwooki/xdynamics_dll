#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xParticleMananger.h"
#include "xdynamics_object/xPlaneObject.h"

xObjectManager* _xom = NULL;

xObjectManager::xObjectManager()
{
	_xom = this;
}

xObjectManager::~xObjectManager()
{
	qDeleteAll(objects);
}

void xObjectManager::addObject(xObject* obj)
{
//	QString qname = QString::fromStdWString(obj->Name().toWString());
	objects[obj->Name()] = obj;
}

xObjectManager* xObjectManager::XOM()
{
	return _xom;
}

xObject* xObjectManager::XObject(std::wstring& _ws)
{
	//QString qname = QString::fromStdWString(ws);
	QString ws = QString::fromStdWString(_ws);
	QStringList keys = objects.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end() || !keys.size())
		return NULL;
	return objects[ws];
}

QMap<QString, xObject*>& xObjectManager::XObjects()
{
	return objects;
}

xLineObject* xObjectManager::CreateLineShapeObject(std::wstring _name, int _xmt)
{
	QString name = QString::fromStdWString(_name);
	xLineObject* xlo = new xLineObject(_name);
	xlo->setMaterialType((xMaterialType)_xmt);
	xMaterial xm = GetMaterialConstant(_xmt);
	xlo->setDensity(xm.density);
	xlo->setYoungs(xm.youngs);
	xlo->setPoisson(xm.poisson);
	objects[name] = xlo;
	return xlo;
}

xPlaneObject* xObjectManager::CreatePlaneShapeObject(std::wstring _name, int _xmt)
{
	QString name = QString::fromStdWString(_name);
	xPlaneObject* xpo = new xPlaneObject(_name);
	xpo->setMaterialType((xMaterialType)_xmt);
	xMaterial xm = GetMaterialConstant(_xmt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	objects[name] = xpo;
	return xpo;
}

xCubeObject* xObjectManager::CreateCubeShapeObject(std::wstring _name, int _xmt)
{
	QString name = QString::fromStdWString(_name);
	xCubeObject* xco = new xCubeObject(_name);
	xco->setMaterialType((xMaterialType)_xmt);
	xMaterial xm = GetMaterialConstant(_xmt);
	xco->setDensity(xm.density);
	xco->setYoungs(xm.youngs);
	xco->setPoisson(xm.poisson);
	objects[name] = xco;
	return xco;
}

xMeshObject* xObjectManager::CreateMeshShapeObject(std::wstring _name, int _xmt)
{
	QString name = QString::fromStdWString(_name);
	xMeshObject* xmo = new xMeshObject(_name);
	xMaterial xm = GetMaterialConstant(_xmt);
	xmo->setDensity(xm.density);
	xmo->setYoungs(xm.youngs);
	xmo->setPoisson(xm.poisson);
	objects[name] = xmo;
	return xmo;
}

xClusterObject * xObjectManager::CreateClusterShapeObject(std::wstring _name, int _xmt)
{
	QString name = QString::fromStdWString(_name);
	xClusterObject* xco = new xClusterObject(_name);
	xMaterial xm = GetMaterialConstant(_xmt);
	xco->setMaterialType((xMaterialType)_xmt);
	xco->setDensity(xm.density);
	xco->setYoungs(xm.youngs);
	xco->setPoisson(xm.poisson);
	objects[name] = xco;
	return xco;
}

void xObjectManager::CreateSPHBoundaryParticles(xParticleManager* xpm)
{
// 	foreach(xObject *xo, XObjects())
// 	{
// 		if (xo->Material() == BOUNDARY)
// 		{
// 			xpm->CreateLineParticle
// 		}
// 	}
}
