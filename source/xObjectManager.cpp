#include "xdynamics_manager/xObjectManager.h"
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

xObject* xObjectManager::XObject(std::string& _ws)
{
	//QString qname = QString::fromStdWString(ws);
	QString ws = QString::fromStdString(_ws);
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

xLineObject* xObjectManager::CreateLineShapeObject(std::string _name, int _xmt)
{
	QString name = QString::fromStdString(_name);
	xLineObject* xlo = new xLineObject(_name);
	xMaterial xm = GetMaterialConstant(_xmt);
	xlo->setDensity(xm.density);
	xlo->setYoungs(xm.youngs);
	xlo->setPoisson(xm.poisson);
	objects[name] = xlo;
	return xlo;
}

xPlaneObject* xObjectManager::CreatePlaneShapeObject(std::string _name, int _xmt)
{
	QString name = QString::fromStdString(_name);
	xPlaneObject* xpo = new xPlaneObject(_name);
	xMaterial xm = GetMaterialConstant(_xmt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	objects[name] = xpo;
	return xpo;
}

xCubeObject* xObjectManager::CreateCubeShapeObject(std::string _name, int _xmt)
{
	QString name = QString::fromStdString(_name);
	xCubeObject* xco = new xCubeObject(_name);
	xMaterial xm = GetMaterialConstant(_xmt);
	xco->setDensity(xm.density);
	xco->setYoungs(xm.youngs);
	xco->setPoisson(xm.poisson);
	objects[name] = xco;
	return xco;
}

xMeshObject* xObjectManager::CreateMeshShapeObject(std::string _name, int _xmt)
{
	QString name = QString::fromStdString(_name);
	xMeshObject* xmo = new xMeshObject(_name);
	xMaterial xm = GetMaterialConstant(_xmt);
	xmo->setDensity(xm.density);
	xmo->setYoungs(xm.youngs);
	xmo->setPoisson(xm.poisson);
	objects[name] = xmo;
	return xmo;
}
