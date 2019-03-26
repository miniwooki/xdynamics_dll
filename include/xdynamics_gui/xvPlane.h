#ifndef XVPLANE_H
#define XVPLANE_H

#include "xvObject.h"

class QTextStream;

class xvPlane : public xvObject
{
public:
	xvPlane();
	xvPlane(QString _name);
	virtual ~xvPlane() { glDeleteLists(glList, 1); }

	virtual void draw(GLenum eMode);
	bool makePlaneGeometry(xPlaneObjectData& d);

private:
	bool define();

	unsigned int glHiList;
	unsigned int glList;
	float width;
	float height;
	vector3f p0;
	vector3f p1;
	vector3f p2;
	vector3f p3;
};

#endif