#ifndef XCOMMANDLINE_H
#define XCOMMANDLINE_H

#include "../xTypes.h"
#include <QString>
#include <QStringList>

class xCommandLine
{
public:
	xCommandLine();
	~xCommandLine();

	bool IsFinished();
	bool IsWrongCommand();
	void SetCurrentAction(int i);
	QString getPassedCommand();
	QString CylinderCommandProcess(QString& com);
	QString CubeCommandProcess(QString& com);
	xCylinderObjectData GetCylinderParameters();
	xCubeObjectData GetCubeParameters();

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
	xCubeObjectData* cube;
};

#endif