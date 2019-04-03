#ifndef XVOBJECT_H
#define XVOBJECT_H

// #ifndef QT_OPENGL_ES_2

// #include <gl/glu.h>
// #endif
#include <gl/glew.h>
#include <QGLWidget>
#include <QString>
#include <QObject>
#include <QMessageBox>
#include <QFile>
#include <QDebug>
#include <QMap>

//#include "model.h"
#include "xvAnimationController.h"
#include "../xTypes.h"
#include "xdynamics_algebra/xAlgebraMath.h"
//#include "types.h"
//#include "algebraMath.h"

class xvObject
{
public:
	enum Type { V_OBJECT = 0, V_MARKER, V_CUBE, V_CYLINDER, V_PLANE, V_POLYGON };
	xvObject();
	xvObject(Type tp, QString _name);
	virtual ~xvObject();

 	Type ObjectType() { return type; }
// // 	void setInitialPosition( ip) { pos0 = ip; }
// // 	void setInitialAngle(VEC3D ia) { ang0 = ia; }
// //	void setCurrentPosition(VEC3D cp) { cpos = cp; }
// //	void setCurrentAngle(VEC3D ca) { cang = ca; }
// 	void animationFrame(VEC3D& p, EPD& ep);
// 	void setResultData(unsigned int n);
// 	void setMaterialType(material_type mt) { m_type = mt; }
// 	void insertResultData(unsigned int i, VEC3D& p, EPD& r);
// 	VEC3D InitialPosition() { return pos0; }
// 	material_type MaterialType() { return m_type; }
 	int ID() { return id; }
 	QString& Name() { return name; }
// 	QString FilePath() { return file_path; }
// 	import_shape_type ImportType() { return ist; }
	void setName(QString n);// { nm = n; }
	void setAngle(float x, float y, float z);
	void setPosition(float x, float y, float z);
// 	void setDisplay(bool _dis) { display = _dis; }
 	QColor Color() { return clr; }
// 	void setColor(color_type ct);
// 	static void msgBox(QString ch, QMessageBox::Icon ic);
// 	void copyCoordinate(GLuint _coord);
// 	void setDrawingMode(GLenum dm) { drawingMode = dm; }
	void setSelected(bool b);//; { isSelected = b; }
// 	context_object_type ViewGeometryObjectType() { return vot; }
 	virtual void draw(GLenum eMode) = 0;
// 
// 	void updateView(VEC3D& _pos, VEC3D& _ang);

// 	double vol;
// 	double mass;
// 	double ixx, iyy, izz;
// 	double ixy, ixz, iyz;

protected:
	int id;
	Type type;
	bool isSelected;
	xvObject* select_cube;
// 	geometry_type g_type;
// 	material_type m_type;
// 	context_object_type vot;
// 	import_shape_type ist;
	QString name;			// object name
	QString file_path;
	GLuint coord;
	GLenum drawingMode;
	bool display;
	QColor clr;
	static int count;
	vector3f pos;
	vector3f ang;
// 	VEC3D pos0;
// 	VEC3D ang0;
// 	VEC3D cpos;
// 	VEC3D cang;
// 	VEC3D* outPos;
// 	EPD* outRot;
};

#endif