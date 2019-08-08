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
#include <QVector>


//#include "model.h"
#include "xvAnimationController.h"
#include "../xTypes.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include "xdynamics_object/xPointMass.h"
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
	void xVertex(float x, float y, float z);
	static void setGlobalMinMax(vector3f v);
	static int xvObjectCount() { return count; }
 	int ID() { return id; }
 	QString& Name() { return name; }
	QString ConnectedMassName() { return connected_mass_name; }
	void setName(QString n);// { nm = n; }
	void setConnectedMassName(QString n);
	void setAngle(float x, float y, float z);
	void setPosition(float x, float y, float z);
	void setBlendAlpha(float v) { blend_alpha = v; }
	void uploadPointMassResults(QString fname);
	void bindPointMassResultsPointer(QVector<xPointMass::pointmass_result>* _pmrs);
 	QColor& Color() { return clr; }
	void setColor(QColor ct) { clr = ct; }
 	void setDrawingMode(GLenum dm) { drawingMode = dm; }
	void setSelected(bool b);
	vector3f Position() { return pos; }

 	virtual void draw(GLenum eMode) = 0;

	static vector3f max_vertex;
	static vector3f min_vertex;

protected:
	int id;
	unsigned int glList;
	Type type;
	bool isSelected;
	bool isBindPmrs;
	xvObject* select_cube;
	QString name;			// object name
	QString connected_mass_name;
	QString file_path;
	GLuint coord;
	GLenum drawingMode;
	bool display;
	QColor clr;
	float blend_alpha;
	static int count;
	vector3f pos;
	vector3f ang;
	vector3f local_min;
	vector3f local_max;
	QVector<xPointMass::pointmass_result>* pmrs;
};

#endif