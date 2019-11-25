#ifndef XVCYLINDER_H
#define XVCYLINDER_H

#include "xvObject.h"

class QTextStream;

class xvCylinder : public xvObject
{
public:
	xvCylinder();
	xvCylinder(QString& _name);

	virtual ~xvCylinder(){ glDeleteLists(glList, 1); }
	virtual void draw(GLenum eMode);
	xCylinderObjectData CubeData() { return data; }
	bool makeCylinderGeometry(xCylinderObjectData& d);
	//bool makeCubeGeometry(QString& _name, geometry_use _tr, material_type _tm, VEC3F& _mp, VEC3F& _sz);

private:
	bool define();
	void setIndexList();
	void setNormalList();
//	unsigned int glList;
	//unsigned int glHiList;
// 	float len;
// 	float r_top, r_bottom;
// 	float p0[3];
// 	float p1[3];
	xCylinderObjectData data;
	vector3d eangle;
};

#endif