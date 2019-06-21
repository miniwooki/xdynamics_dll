#include "xViewExporter.h"
//#include <QtCore/QFile>

std::wfstream *of = NULL;

xViewExporter::xViewExporter()
{

}

xViewExporter::~xViewExporter()
{

}

void xViewExporter::Open(std::wstring path)
{
	of = new std::wfstream;
	of->open(path, std::ios::out | std::ios::binary);
	if (of->is_open())
	{
		int ver = VERSION_NUMBER;
		of->write((wchar_t*)&ver, sizeof(int));
	}
}

void xViewExporter::Write(wchar_t* c, unsigned int sz)
{
	if (of)
	{
		if (of->is_open())
			of->write(c, sz);
	}
}

void xViewExporter::Close()
{
	if (of)
	{
		of->close();
		delete of;
	}
}
