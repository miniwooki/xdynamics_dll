#ifndef XVMARKER_H
#define XVMARKER_H

#include "xvObject.h"

class QTextStream;

class xvMarker : public xvObject
{
public:
	xvMarker();
	xvMarker(QString& _name, bool mcf = true);
	//vmarker(QTextStream& in);
	virtual ~xvMarker();

	virtual void draw(GLenum eMode);

	//void setMarkerScale
	bool define(float x, float y, float z, bool isdefine_text = false);
	//void setAttchedMass(bool b) { isAttachMass = b; }
	//void setAttachObject(QString o) { attachObject = o; }
	//void setMarkerScaleFlag(bool b) { markerScaleFlag = b; }
	void setMarkerScale(float sc) { scale = sc; }
	//bool makeCubeGeometry(QTextStream& in);
	//bool makeCubeGeometry(QString& _name, geometry_use _tr, material_type _tm, VEC3F& _mp, VEC3F& _sz);

private:
	bool isAttachMass;
	QString attachObject;
	bool markerScaleFlag;
	float scale;
	unsigned int glList;
	//static float icon_scale;
	//unsigned int glHiList;
	//	float loc[3];
	// 	int indice[24];
	// 	float vertice[24];
	// 	float normal[18];
};

#endif