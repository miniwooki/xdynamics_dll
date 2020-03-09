#ifndef XOBJECTMANAGER_H
#define XOBJECTMANAGER_H

#include "xModel.h"
#include "xdynamics_decl.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_object/xLineObject.h"
#include "xdynamics_object/xCubeObject.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xMeshObject.h"
#include "xdynamics_object/xCylinderObject.h"
#include "xdynamics_object/xClusterObject.h"
#include "xdynamics_object/xMassSpringTireModel.h"

#include "xmap.hpp"
#include <map>

class xParticleManager;

class XDYNAMICS_API xObjectManager
{
public:
	xObjectManager();
	~xObjectManager();

	void addObject(xObject* obj);
	static xObjectManager* XOM();
	xObject* XObject(std::string& ws);
	xmap<xstring, xObject*>& XObjects();
	map<std::string, xObject*> XObjectsToSTD();
	map<std::string, xClusterObject*> XClusterObjects();
	xMassSpringTireModel* GeneralSpringDamper();
	xPointMass* setMovingConstantMovingVelocity(std::string _name, double* v);
	void UpdateMovingObjects(double ct);
	void SaveResultCompulsionMovingObjects(double ct);
	xLineObject* CreateLineShapeObject(std::string _name, int _xmt);
	xPlaneObject* CreatePlaneShapeObject(std::string _name, int _xmt);
	xCubeObject* CreateCubeShapeObject(std::string _name, int _xmt);
	xMeshObject* CreateMeshShapeObject(std::string _name, int _xmt);
	xClusterObject* CreateClusterShapeObject(std::string _name, int _xmt);
	xCylinderObject* CreateCylinderShapeObject(std::string _name, int _xmt);
	xMassSpringTireModel* CreateGeneralSpringDamper(std::string _name, std::string filepath);
	bool InsertClusterShapeObject(xClusterObject*);

	void CreateSPHBoundaryParticles(xParticleManager* xpm);
	xmap<xstring, xObject*>& CompulsionMovingObjects();

private:
	xmap<xstring, xObject*> objects;
	xmap<xstring, xObject*> compulsion_moving_objects;
	xMassSpringTireModel* general_sd;
};

#endif