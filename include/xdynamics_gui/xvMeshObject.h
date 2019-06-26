#ifndef VPOLYGON_H
#define VPOLYGON_H

#include "xvGlew.h"
#include "xvObject.h"
#include "xvAnimationController.h"
//#include "vobject.h"

#include <QList>

class QTextStream;

class xvMeshObject : public xvGlew, public xvObject
{
public:
	xvMeshObject();
	xvMeshObject(QString& _name);
	virtual ~xvMeshObject();
	virtual void draw(GLenum eMode);
	void defineMeshObject(unsigned int nt, double* v, double* n);
	unsigned int NumTriangles() { return ntriangle; }
	QString GenerateFitSphereFile(float ft);
	vector4f FitSphereToTriangle(vector3f& P, vector3f& Q, vector3f& R, float ft);

private:
	void _drawPolygons();

	unsigned int ntriangle;
	unsigned int m_vertex_vbo;
	unsigned int m_normal_vbo;
	float *vertexList;
	float *normalList;
	float *texture;
	float *colors;

	shaderProgram program;
};

#endif