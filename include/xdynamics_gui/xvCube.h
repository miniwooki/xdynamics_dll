#ifndef XVCUBE_H
#define XVCUBE_H

#include "xvObject.h"

class QTextStream;

class xvCube : public xvObject
{
public:
	xvCube();
	xvCube(QString& _name);
	
	virtual ~xvCube(){ glDeleteLists(glList, 1); }
	virtual void draw(GLenum eMode);
	xCubeObjectData CubeData() { return data; }
	bool makeCubeGeometry(xCubeObjectData& d);
	//bool makeCubeGeometry(QString& _name, geometry_use _tr, material_type _tm, VEC3F& _mp, VEC3F& _sz);

private:
	bool define();
	void setIndexList();
	void setNormalList();
	unsigned int glList;
	unsigned int glHiList;
	//float origin[3];
	int indice[24];
	float vertice[24];
	float normal[18];
	xCubeObjectData data;
};

#endif