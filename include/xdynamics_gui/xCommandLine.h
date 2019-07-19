#ifndef XCOMMANDLINE_H
#define XCOMMANDLINE_H

#include "../xTypes.h"
#include <QString>
#include <QStringList>

class xvObject;

class xCommandLine
{
public:
	xCommandLine();
	~xCommandLine();

	bool IsFinished();
	bool IsWrongCommand();
	void SetCurrentAction(int i);
	void SetCurrentObject(xvObject* vxo);
	QString getPassedCommand();
	QString CylinderCommandProcess(QString& com);
	QString CubeCommandProcess(QString& com);
	QString MeshObjectCommandProcess(QString& com);
	xCylinderObjectData GetCylinderParameters();
	xCubeObjectData GetCubeParameters();
	xvObject* GetCurrentObject();

private:
	bool is_finished;
	bool isOnCommand;
	bool isWrongCommand;
	int sz;
	int cidx;
	int cstep;
	int caction;
	QStringList sList;
	int current_log_index;
	QStringList logs;
	xCylinderObjectData* cylinder;
	xvObject* xobj;
	xCubeObjectData* cube;
};

#endif