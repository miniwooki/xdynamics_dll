#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xParticleMananger.h"
#include "xdynamics_object/xPlaneObject.h"

xObjectManager* _xom = NULL;

xObjectManager::xObjectManager()
	: general_sd(nullptr)
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

xGeneralSpringDamper * xObjectManager::GeneralSpringDamper()
{
	return general_sd;
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

xGeneralSpringDamper * xObjectManager::CreateGeneralSpringDamper(std::string _name, std::string filepath)
{
	std::fstream fs;
	fs.open(filepath, ios_base::in);
	if (fs.is_open()) {
		std::string ch;
		while (!fs.eof()) {
			//std::string name = xUtilityFunctions::GetFileName(filepath.c_str());
			general_sd = new xGeneralSpringDamper(_name);

			fs >> ch;
			if (ch == "point2spring") {
				unsigned int np2s = 0;
				fs >> np2s;
				for (unsigned int i = 0; i < np2s; i++) {
					std::string p2s_name;
					fs >> p2s_name;
					xPoint2Spring *p2s = new xPoint2Spring(p2s_name);
					objects.insert(p2s_name, p2s);
					unsigned int i0, i1;
					vector3d sp0, sp1;
					double k, c, len;
					fs >> i0 >> i1 >> sp0.x >> sp0.y >> sp0.z >> sp1.x >> sp1.y >> sp1.z
						>> k >> c >> len;
					p2s->SetSpringDamperData(i0, i1, sp0, sp1, k, c, len);
					general_sd->appendPoint2Spring(p2s);
				}
			}
			else if (ch == "attach_point") {
				unsigned int nap = 0;
				fs >> nap;
				if (nap)
					general_sd->allocAttachPoint(nap);
				for (unsigned int i = 0; i < nap; i++) {
					std::string ap_name;
					fs >> ap_name;
					xAttachPoint xap = { 0, };
					xap.body = ap_name.c_str();
					fs >> xap.id >> xap.rx >> xap.ry >> xap.rz;
					general_sd->appendAttachPoint(i, xap);
				}
			}
			else if (ch == "rotational_spring") {
				unsigned int nrs = 0;
				fs >> nrs;
				if (nrs)
					general_sd->allocRotationalSpring(nrs);
				for (unsigned int i = 0; i < nrs; i++) {
					std::string rs_name;
					fs >> rs_name;
					xRotationalSpringData rsd = { 0, };
					rsd.body = rs_name.c_str();
					fs >> rsd.id >> rsd.rx >> rsd.ry >> rsd.rz >> rsd.k >> rsd.c;
					general_sd->appendRotationalSpring(i, rsd);
				}				
			}
		}
	}
	return general_sd;
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
