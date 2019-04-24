#ifndef XOBJECTMANAGER_H
#define XOBJECTMANAGER_H

#include "xModel.h"
#include "xdynamics_decl.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_object/xLineObject.h"
#include "xdynamics_object/xCubeObject.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xMeshObject.h"

class XDYNAMICS_API xObjectManager
{
public:
	xObjectManager();
	~xObjectManager();

	void addObject(xObject* obj);
	static xObjectManager* XOM();
	xObject* XObject(std::string& ws);
	QMap<QString, xObject*>& XObjects();

	xLineObject* CreateLineShapeObject(std::string _name, int _xmt);
	xPlaneObject* CreatePlaneShapeObject(std::string _name, int _xmt);
	xCubeObject* CreateCubeShapeObject(std::string _name, int _xmt);
	xMeshObject* CreateMeshShapeObject(std::string _name, int _xmt);

private:
	QMap<QString, xObject*> objects;
};

#endif