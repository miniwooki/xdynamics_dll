#ifndef XRESULTMANAGER_H
#define XRESULTMANAGER_H

#include "xdynamics_decl.h"
#include <QtCore/QString>

class XDYNAMICS_API xResultManager
{
public:
	xResultManager();
	~xResultManager();

	void xRun(const std::wstring _cpath, const std::wstring _cname);

private:
	void setCurrentPath(std::wstring new_path);
	void setCurrentName(std::wstring new_name);

	void ExportBPM2TXT(std::wstring& file_name);
	void ExportBKC2TXT(std::wstring& file_name);

	int Execute0(wchar_t *d);
	int Execute1(wchar_t *d);
	int Execute2(wchar_t *d);

	QString cur_path;// wchar_t cur_path[PATH_BUFFER_SIZE];
	QString cur_name;// wchar_t cur_name[NAME_BUFFER_SIZE];
};

#endif