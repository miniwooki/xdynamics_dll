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

	void Open(std::wstring path);
	void Write(wchar_t* c, unsigned int sz);
	void Close();
	
private:
	QString path;
};

#endif