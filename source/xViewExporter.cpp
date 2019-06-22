#include "xViewExporter.h"
//#include <QtCore/QFile>

std::fstream *of = NULL;

xViewExporter::xViewExporter()
{

}

xViewExporter::~xViewExporter()
{

}

void xViewExporter::Open(std::string path)
{
	of = new std::fstream;
	of->open(path, std::ios::out | std::ios::binary);
	if (of->is_open())
	{
		int ver = VERSION_NUMBER;
		of->write((char*)&ver, sizeof(int));
	}
}

void xViewExporter::Write(char* c, unsigned int sz)
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
