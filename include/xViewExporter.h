#ifndef XVIEWEXPORTER_H
#define XVIEWEXPORTER_H

#include "xdynamics_decl.h"
#include <fstream>
#include <QtCore/QString>

class XDYNAMICS_API xViewExporter
{
public:
	xViewExporter();
	~xViewExporter();

	void Open(std::string path);
	void Write(char* c, unsigned int sz);
	void Close();
	
private:
	QString path;
};

#endif