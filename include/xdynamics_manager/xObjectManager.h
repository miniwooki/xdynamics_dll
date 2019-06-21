#ifndef XOBJECTMANAGER_H
#define XOBJECTMANAGER_H

#include "xModel.h"
#include "xdynamics_decl.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_object/xLineObject.h"
#include "xdynamics_object/xCubeObject.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xMeshObject.h"
#include "xdynamics_object/xClusterObject.h"

class xParticleManager;

class XDYNAMICS_API xObjectManager
{
public:
	xObjectManager();
	~xObjectManager();

	void addObject(xObject* obj);
	static xObjectManager* XOM();
	xObject* XObject(std::wstring& ws);
	QMap<QString, xObject*>& XObjects();

	xLineObject* CreateLineShapeObject(std::wstring _name, int _xmt);
	xPlaneObject* CreatePlaneShapeObject(std::wstring _name, int _xmt);
	xCubeObject* CreateCubeShapeObject(std::wstring _name, int _xmt);
	xMeshObject* CreateMeshShapeObject(std::wstring _name, int _xmt);
	xClusterObject* CreateClusterShapeObject(std::wstring _name, int _xmt);

	void CreateSPHBoundaryParticles(xParticleManager* xpm);

private:
	QMap<QString, xObject*> objects;
};

#endif