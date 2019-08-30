#ifndef XRESULTMANAGER_H
#define XRESULTMANAGER_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xPointMass.h"
#include <QtCore/QString>

class XDYNAMICS_API xResultManager
{
public:
	xResultManager();
	~xResultManager();

	void xRun(const std::string _cpath, const std::string _cname);
	static std::string ExportPointMassResult2TXT(std::string name, QVector<xPointMass::pointmass_result>* rst);

private:
	void setCurrentPath(std::string new_path);
	void setCurrentName(std::string new_name);

	void ExportBPM2TXT(std::string& file_name);
	void ExportBKC2TXT(std::string& file_name);

	int Execute0(char *d);
	int Execute1(char *d);
	int Execute2(char *d);

	QString cur_path;// char cur_path[PATH_BUFFER_SIZE];
	QString cur_name;// char cur_name[NAME_BUFFER_SIZE];
};

#endif