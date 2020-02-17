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
	if (objects.size()) objects.delete_all();
	if (compulsion_moving_objects.size()) compulsion_moving_objects.delete_all();
}

void xObjectManager::addObject(xObject* obj)
{
//	QString qname = QString::fromStdString(obj->Name().toWString());
	objects.insert(obj->Name(), obj);
}

xObjectManager* xObjectManager::XOM()
{
	return _xom;
}

xObject* xObjectManager::XObject(std::string& _ws)
{
	//QString qname = QString::fromStdString(ws);
	//QString ws = QString::fromStdString(_ws);
	//QStringList keys = objects.keys();
	if (!objects.size())
		return NULL;
	xmap<xstring, xObject*>::iterator it = objects.find(_ws);//QStringList::const_iterator it = qFind(keys, ws);
	if (it == objects.end())
		return NULL;
	return it.value();// objects[ws];
}

xmap<xstring, xObject*>& xObjectManager::XObjects()
{
	return objects;
}

map<std::string, xObject*> xObjectManager::XObjectsToSTD()
{
	xmap<xstring, xObject*>::iterator it = objects.begin();
	map<std::string, xObject*> ret;
	while (it.has_next()) {
		ret[it.key().toStdString()] = it.value();
	}
	return ret;
}

map<std::string, xClusterObject*> xObjectManager::XClusterObjects()
{
	xmap<xstring, xObject*>::iterator it = objects.begin();
	map<std::string, xClusterObject*> ret;
	while (it.has_next()) {
		if(it.value()->Shape() == CLUSTER_SHAPE)
			ret[it.key().toStdString()] = dynamic_cast<xClusterObject*>(it.value());
		it.next();
	}
	return ret;
}

xPointMass * xObjectManager::setMovingConstantMovingVelocity(std::string _name, double* v)
{
	xPointMass* xpm = dynamic_cast<xPointMass*>(XObject(_name));
	if (xpm)
		xpm->setMovingConstantMovingVelocity(new_vector3d(v[0], v[1], v[2]));
	compulsion_moving_objects.insert(_name, xpm);
	return xpm;
}

void xObjectManager::UpdateMovingObjects(double ct)
{
	xmap<xstring, xObject*>::iterator it = compulsion_moving_objects.begin();
	for (; it != compulsion_moving_objects.end(); it.next())// foreach(xObject* xo, compulsion_moving_objects)
	{
		xPointMass* xpm = dynamic_cast<xPointMass*>(it.value());
		xpm->UpdateByCompulsion(ct);
		xpm->setZeroAllForce();
	}
}

void xObjectManager::SaveResultCompulsionMovingObjects(double ct)
{
	xmap<xstring, xObject*>::iterator it = compulsion_moving_objects.begin();
	for (; it != compulsion_moving_objects.end(); it.next())
	{
		xPointMass* xpm = dynamic_cast<xPointMass*>(it.value());
		xpm->SaveStepResult(ct);
	}
}

xLineObject* xObjectManager::CreateLineShapeObject(std::string _name, int _xmt)
{
	xstring name = _name;
	xLineObject* xlo = new xLineObject(_name);
	xlo->setMaterialType((xMaterialType)_xmt);
	/*xMaterial xm = GetMaterialConstant(_xmt);
	xlo->setDensity(xm.density);
	xlo->setYoungs(xm.youngs);
	xlo->setPoisson(xm.poisson);*/
	objects.insert(name, xlo);
	return xlo;
}

xPlaneObject* xObjectManager::CreatePlaneShapeObject(std::string _name, int _xmt)
{
	xstring name = _name;
	xPlaneObject* xpo = new xPlaneObject(_name);
	xpo->setMaterialType((xMaterialType)_xmt);
	//xMaterial xm = GetMaterialConstant(_xmt);
	//xpo->setDensity(xm.density);
	//xpo->setYoungs(xm.youngs);
	//xpo->setPoisson(xm.poisson);
	objects.insert(name, xpo);
	return xpo;
}

xCubeObject* xObjectManager::CreateCubeShapeObject(std::string _name, int _xmt)
{
	xstring name = _name;
	xCubeObject* xco = new xCubeObject(_name);
	xco->setMaterialType((xMaterialType)_xmt);
	/*xMaterial xm = GetMaterialConstant(_xmt);
	xco->setDensity(xm.density);
	xco->setYoungs(xm.youngs);
	xco->setPoisson(xm.poisson);*/
	objects.insert(name, xco);
	return xco;
}

xMeshObject* xObjectManager::CreateMeshShapeObject(std::string _name, int _xmt)
{
	xstring name = _name;
	xMeshObject* xmo = new xMeshObject(_name);
	xmo->setMaterialType((xMaterialType)_xmt);
	//xMaterial xm = GetMaterialConstant(_xmt);
	/*xmo->setDensity(xm.density);
	xmo->setYoungs(xm.youngs);
	xmo->setPoisson(xm.poisson);*/
	objects.insert(name, xmo);
	return xmo;
}

xClusterObject * xObjectManager::CreateClusterShapeObject(std::string _name, int _xmt)
{
	xstring name = _name;
	xClusterObject* xco = new xClusterObject(_name);
	//xMaterial xm = GetMaterialConstant(_xmt);
	xco->setMaterialType((xMaterialType)_xmt);
	//xco->setDensity(xm.density);
	//xco->setYoungs(xm.youngs);
	//xco->setPoisson(xm.poisson);
	objects.insert(name, xco);
	return xco;
}

xCylinderObject* xObjectManager::CreateCylinderShapeObject(std::string _name, int _xmt)
{
	xstring name = _name;
	xCylinderObject* xco = new xCylinderObject(_name);
	//xMaterial xm = GetMaterialConstant(_xmt);
	xco->setMaterialType((xMaterialType)_xmt);
	//xco->setDensity(xm.density);
	//xco->setYoungs(xm.youngs);
	//xco->setPoisson(xm.poisson);
	objects.insert(name, xco);
	return xco;
}

bool xObjectManager::InsertClusterShapeObject(xClusterObject *cobj)
{
	if (objects.find(cobj->Name()) == objects.end()) {
		objects.insert(cobj->Name(), cobj);
		return true;
	}
	return false;
}

void xObjectManager::CreateSPHBoundaryParticles(xParticleManager* xpm)
{

}

xmap<xstring, xObject*>& xObjectManager::CompulsionMovingObjects()
{
	return compulsion_moving_objects;
}
