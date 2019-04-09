#ifndef XMESHOBJECT_H
#define XMESHOBJECT_H

#include "xdynamics_object/xPointMass.h"
#include <QtCore/QList>

class XDYNAMICS_API xMeshObject : public xPointMass
{
	typedef struct
	{
		double rad;
		vector3d p, q, r, n;
	}triangle_info;
public:
	xMeshObject();
	xMeshObject(std::string file);
	xMeshObject(const xMeshObject& mesh);
	virtual ~xMeshObject();

	bool define(xImportShapeType t, vector3d& loc, int ntriangle, double* vList, unsigned int *iList);
	int DefineShapeFromFile(vector3d & loc, std::string f);
	void updateDeviceFromHost();

	QString meshDataFile() const;// { return filePath; }
	double maxRadius() const;// { return maxRadii; }
	unsigned int NumTriangle() const;// { return ntriangle; }
	double* VertexList();// { return vertexList; }
	double* NormalList();// { return indexList; }
	vector3d MaxPoint() const;
	vector3d MinPoint() const;
	void splitTriangles(double to);

private:
	void _fromSTLASCII(int _ntriangle, double* vList, vector3d& loc);
	QList<triangle_info> _splitTriangle(triangle_info& ti, double to);

private:
	unsigned int ntriangle;
	double maxRadii;
	double *vertexList;
	double *normalList;
	vector3d max_point;
	vector3d min_point;
	//unsigned int *indexList;

	QString filePath;
};

#endif