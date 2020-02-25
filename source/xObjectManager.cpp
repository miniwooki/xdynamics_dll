#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xParticleMananger.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xRotationSpringDamperForce.h"
#include "xdynamics_object/xSpringDamperForce.h"
#include "xdynamics_object/xParticle.h"

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
		general_sd = new xGeneralSpringDamper(_name);
		while (!fs.eof()) {
			fs >> ch;
			if (ch == "mass_particle_translational_sping")
			{
				unsigned int nmp = 0;
				fs >> nmp;
				for (unsigned int i = 0; i < nmp; i++) {
					std::string ts_name;
					std::string bd_name;
					std::string pt_name;
					fs >> ts_name >> bd_name >> pt_name;
					xTSDAData tsd = { 0, };
					fs >> tsd.spix >> tsd.spiy >> tsd.spiz
						>> tsd.spjx >> tsd.spjy >> tsd.spjz
						>> tsd.k >> tsd.c;
					xObject* body = objects.find(bd_name).value();
					xParticle* pt = new xParticle(pt_name);
					objects.insert(pt_name, pt);
					unsigned int pt_id = atoi(pt_name.substr(1, pt_name.size() - 1).c_str());
					pt->setIndex(pt_id);
					pt->setPosition(tsd.spjx, tsd.spjy, tsd.spjz);
					xSpringDamperForce* ts = new xSpringDamperForce(ts_name);
					ts->SetupDataFromStructure(
						dynamic_cast<xPointMass*>(body), 
						pt,
						tsd);
				}
			}
			/*else if (ch == "point2spring") {
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
			}*/
			else if (ch == "attach_point") {
				unsigned int nap = 0;
				fs >> nap;
				if (nap)
					general_sd->allocAttachPoint(nap);
				for (unsigned int i = 0; i < nap; i++) {
					std::string ap_name;
					fs >> ap_name;
					xObject *body = objects.find(ap_name).value();
					xAttachPoint xap = { 0, };
					xap.body = body->xCharName();
					fs >> xap.id >> xap.rx >> xap.ry >> xap.rz;
					general_sd->appendAttachPoint(i, xap);
				}
			}
			else if (ch == "rotational_spring") {
				unsigned int nrs = 0;
				fs >> nrs;
				for (unsigned int i = 0; i < nrs; i++) {
					std::string rs_name;
					std::string bd0_name;
					std::string bd1_name;
					fs >> rs_name;
					fs >> bd0_name;
					fs >> bd1_name;
					xObject* body0 = objects.find(bd0_name).value();
					xObject* body1 = objects.find(bd1_name).value();// general_sd->Point2Spring(bd1_name);
					//xRotationalSpringData rsd = { 0, };
					xRSDAData rsd = { 0, };
					//rsd.body = body->xCharName();
					//rsd.p2s = ps->xCharName();
					fs >> rsd.lx >> rsd.ly >> rsd.lz
						>> rsd.fix >> rsd.fiy >> rsd.fiz
						>> rsd.gix >> rsd.giy >> rsd.giz
						>> rsd.fjx >> rsd.fjy >> rsd.fjz
						>> rsd.gjx >> rsd.gjy >> rsd.gjz
						>> rsd.k >> rsd.c;
					xRotationSpringDamperForce* rs = new xRotationSpringDamperForce(rs_name);
					rs->SetupDataFromStructure(
						dynamic_cast<xPointMass*>(body0),
						dynamic_cast<xPointMass*>(body1),
						rsd);
					//general_sd->appendRotationalSpring(i, rsd);

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
