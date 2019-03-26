#ifndef XMODEL_H
#define XMODEL_H

#include "xdynamics_decl.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "xLog.h"
#include <fstream>
#include <QtCore/QString>
#include <QtCore/QMap>

#define RESULT_FILE_IDENTIFIER 11472162

class xPointMass;

class XDYNAMICS_API xModel
{
public:
	enum unit_type{ MMKS = 0, MKS };
	enum angle_type{ EULER_ANGLE = 6, EULER_PARAMETERS = 7 };

	xModel();
	xModel(QString _name);
	virtual ~xModel();

	static int OneDOF();
	static xPointMass* Ground();
	static void setModelName(const QString n);
	static void setModelPath(const QString p);
	static void setGravity(double g, int d);
	static void launchLogSystem(std::string lpath);
	//static void 

	//static bool isSinglePrecision;
	static xPointMass* ground;
	static angle_type angle;
	static unit_type unit;
	static QString name;
	static QString path;
	static vector3d gravity;
	//static resultStorage *rs;
	//static int count;
};

#endif