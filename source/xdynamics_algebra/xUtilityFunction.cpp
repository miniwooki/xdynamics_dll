//#include "stdafx.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "xdynamics_algebra/xAlgebraMath.h"
/*#include "boost/filesystem.hpp"*/
#include <ctime>
#include <filesystem>
#include <QtCore/QDir>
#include <QtCore/QStringList>

//using namespace boost::filesystem;

std::string xUtilityFunctions::xstring(int v)
{
	return QString("%1").arg(v).toStdString();
}

std::string xUtilityFunctions::xstring(unsigned int v)
{
	return QString("%1").arg(v).toStdString();
}

std::string xUtilityFunctions::xstring(double v)
{
	return QString("%1").arg(v).toStdString();
}

void xUtilityFunctions::CreateDirectory(const wchar_t* _path)
{
	QString path = QString::fromStdWString(_path);
	if (!QDir(path).exists())
		QDir().mkdir(path);
// 	try
// 	{
// 		boost::filesystem::path p(_path);
// 		if (!boost::filesystem::exists(p))
// 			boost::filesystem::create_directory(p);
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::cout << ex.what() << std::endl;
// 		throw;
// 	}
}

std::string xUtilityFunctions::xstring(std::wstring v)
{
	return QString::fromStdWString(v).toStdString();
}

QString xUtilityFunctions::GetFileName(const wchar_t* pn)
{
	QString path = QString::fromStdWString(pn);
	int begin = path.lastIndexOf('/');
	int end = path.lastIndexOf('.');
	return path.mid(begin+1, end - begin-1);
// 	QString b;
// 	try
// 	{
// 		boost::filesystem::path p(pn);
// 		b = boost::filesystem::basename(p).c_str();
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::cout << ex.what() << std::endl;
// 		throw;
// 	}
// 	return b;
}

QString xUtilityFunctions::FileExtension(const wchar_t* f)
{
	QString path = QString::fromStdWString(f);
	int begin = path.lastIndexOf('.');
	return path.mid(begin);
// 	
// 	try
// 	{
// 		boost::filesystem::path p(f);
// 		b = boost::filesystem::extension(p).c_str();
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::wcout << ex.what() << std::endl;
// 		throw;
// 	}
// 	return b;
}

bool xUtilityFunctions::ExistFile(const wchar_t* n)
{
	QString path = QString::fromStdWString(n);
	return QFile().exists(path);
// 	bool b = false;
// 	try
// 	{
// 		boost::filesystem::path p(n);
// 		b = boost::filesystem::exists(p);
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::wcout << ex.what() << std::endl;
// 		throw;
// 	}
// 	return b;
}

void xUtilityFunctions::DirectoryFileList(const wchar_t* _path)
{
	QString path = QString::fromStdWString(_path);
	QDir d(path);
	QStringList ls = d.entryList();
	foreach(QString s, ls)
	{
		std::cout << "     " << s.toLocal8Bit().data() << std::endl;
	}
// 	try
// 	{
// 		boost::filesystem::path p(_path);
// 		if (boost::filesystem::exists(p))
// 		{
// 			std::wcout << std::endl << "File list of " << _path << std::endl;
// 			for (auto i = directory_iterator(p); i != directory_iterator(); i++)
// 			{
// 				if (!is_directory(i->path()))
// 				{
// 					std::wcout << "     " << i->path().filename().string() << std::endl;
// 				}
// 			}
// 		}
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::wcout << ex.what() << std::endl;
// 		throw;
// 	}
}

std::wstring xUtilityFunctions::Multibyte2WString(const char* c)
{
	int len = (int)strlen(c);
	wchar_t* wcArr = NULL;// new wchar_t[len];// (wchar_t*)calloc(1, len);
	len = MultiByteToWideChar(CP_ACP, 0, c, len, NULL, NULL);
	wcArr = SysAllocStringLen(NULL, len);
	MultiByteToWideChar(CP_ACP, 0, c, len, wcArr, len);
	wstring cstr = wcArr;
	SysFreeString(wcArr);// [] wcArr;
	return cstr;
}

std::string xUtilityFunctions::WideChar2String(const wchar_t* wc)
{
	int wLen;
	char* cArr;

	wLen = (int)wcslen(wc);
	cArr = (char*)calloc(1, wLen * 2 + 1);
	WideCharToMultiByte(CP_ACP, 0, wc, wLen, cArr, wLen * 2, 0, 0);
	string cstr = cArr;
	delete cArr;
	return cstr;
}

std::wstring xUtilityFunctions::GetDateTimeFormat(const char* format, int nseg)
{
	time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);
	rawtime += nseg;
	localtime_s(&timeinfo, &rawtime);
	char bufftime[64];
	strftime(bufftime, 64, format, &timeinfo);
	return Multibyte2WString(bufftime);
}

int xUtilityFunctions::FindNumString(const string& s, const char* c)
{
	basic_string<char>::size_type start = 0, end;
	static const basic_string<char>::size_type npos = -1;
	int ns = 0;
	int len = (int)strlen(c);
	while (1)
	{
		end = s.find(c, start);
		start = end + len;
		ns++;
		if (end == npos)
			break;
	}
	return ns;
}

void xUtilityFunctions::xsplit(const std::wstring& s, const char* c, int n, int* data)
{
	QString d = QString::fromStdWString(s);
	QStringList ds = d.split(c);
	for (int i = 0; i < n; i++)
		data[i] = ds.at(i).toInt();
}

void xUtilityFunctions::xsplit(const std::wstring& s, const char* c, vector2i& data)
{
	QString d = QString::fromStdWString(s);
	QStringList ds = d.split(c);
	data.x = ds.at(0).toDouble();
	data.y = ds.at(1).toDouble();
}

void xUtilityFunctions::xsplit(const std::wstring& s, const char* c, int n, double* data)
{
	QString d = QString::fromStdWString(s);
	QStringList ds = d.split(c);
	for (int i = 0; i < n; i++)
		data[i] = ds.at(i).toDouble();
// 	basic_string<char>::size_type start = 0, end;
// 	static const basic_string<char>::size_type npos = -1;
// 	int ns = 0;
// 	int len = (int)strlen(c);
// 	while (1)
// 	{
// 		end = s.find(c, start);
// 		string cs = s.substr(start, end - start);
// 		data[ns] = atof(cs.c_str());
// 		start = end + len;
// 		ns++;
// 		if (end == npos)
// 			break;
// 	}
}

void xUtilityFunctions::xsplit(const wchar_t* wc, const char* c, int n, int* data)
{
	QString d = toWideCharToQString(wc);
	QStringList ds = d.split(c);
	for (int i = 0; i < n; i++)
		data[i] = ds.at(i).toInt();
// 	string s = WideChar2String(wc);
// 	basic_string<char>::size_type start = 0, end;
// 	static const basic_string<char>::size_type npos = -1;
// 	int ns = 0;
// 	int len = (int)strlen(c);
// 	while (1)
// 	{
// 		end = s.find(c, start);
// 		string cs = s.substr(start, end - start);
// 		data[ns] = atoi(cs.c_str());
// 		start = end + len;
// 		ns++;
// 		if (end == npos)
// 			break;
// 	}
}

void xUtilityFunctions::xsplit(const wchar_t* wc, const char* c, int n, double* data)
{
	QString d = toWideCharToQString(wc);
	QStringList ds = d.split(c);
	for (int i = 0; i < n; i++)
		data[i] = ds.at(i).toDouble();
// 	string s = WideChar2String(wc);
// 	basic_string<char>::size_type start = 0, end;
// 	static const basic_string<char>::size_type npos = -1;
// 	int ns = 0;
// 	int len = (int)strlen(c);
// 	while (1)
// 	{
// 		end = s.find(c, start);
// 		string cs = s.substr(start, end - start);
// 		data[ns] = atof(cs.c_str());
// 		start = end + len;
// 		ns++;
// 		if (end == npos)
// 			break;
// 	}
}

unsigned int xUtilityFunctions::xsplitn(const wchar_t* s, const char* c)
{
	QString d = toWideCharToQString(s);
	return d.split(c).size();
}

void xUtilityFunctions::xsplit(const wchar_t* wc, const char* c, int n, std::wstring* data)
{
	QString d = toWideCharToQString(wc);
	QStringList ds = d.split(c);
	for (int i = 0; i < n; i++)
		data[i] = ds.at(i).toStdWString();
	// 	string s = WideChar2String(wc);
	// 	basic_string<char>::size_type start = 0, end;
	// 	static const basic_string<char>::size_type npos = -1;
	// 	int ns = 0;
	// 	int len = (int)strlen(c);
	// 	while (1)
	// 	{
	// 		end = s.find(c, start);
	// 		string cs = s.substr(start, end - start);
	// 		data[ns] = atof(cs.c_str());
	// 		start = end + len;
	// 		ns++;
	// 		if (end == npos)
	// 			break;
	// 	}
}

// void xUtilityFunctions::Split2WString(const wchar_t* wc, const wchar_t* c, int& n, wstring* data)
// {
// 	n = 0;
// 	wstring s = wc;
// 	basic_string<wchar_t>::size_type start = 0, end;
// 	static const basic_string<wchar_t>::size_type npos = -1;
// 	int ns = 0;
// 	int len = (int)wcslen(c);
// 	while (1)
// 	{
// 		end = s.find(c, start);
// 		//wstring cs = 
// 		data[ns] = s.substr(start, end - start);// atof(cs.c_str());
// 		start = end + len;
// 		ns++;
// 		if (end == npos)
// 			break;
// 	}
// 	n = ns;
// }

double xUtilityFunctions::AngleCorrection(double d, double th)
{
	unsigned int n = 0;
	unsigned int b = 0;
	unsigned int e = 1;
	unsigned int c = 0;
	double v = 0.0;
	if (th == 0.0)
		return th;
	while (1)
	{
		double _b = b * M_PI;
		double _e = e * M_PI;
		if (d > _b && d < _e)
		{
			if (!c)
				return th;
			n = 2 * (e / 2);
			unsigned int m = c % 2;
			if (m)
				v = n * M_PI - th;
			else
				v = n * M_PI + th;
			return v;
		}
		else
		{
			b++;
			e++;
			c++;
		}
	}
	return 0.0;
}

double xUtilityFunctions::SignedVolumeOfTriangle(vector3d& v1, vector3d& v2, vector3d& v3)
{
	double v321 = v3.x*v2.y*v1.z;
	double v231 = v2.x*v3.y*v1.z;
	double v312 = v3.x*v1.y*v2.z;
	double v132 = v1.x*v3.y*v2.z;
	double v213 = v2.x*v1.y*v3.z;
	double v123 = v1.x*v2.y*v3.z;
	return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123);
}

vector3d xUtilityFunctions::CenterOfTriangle(vector3d& P, vector3d& Q, vector3d& R)
{
	vector3d V = Q - P;
	vector3d W = R - P;
	vector3d N = cross(V, W);
	N = N / length(N);// .length();
	vector3d M1 = (Q + P) / 2;
	vector3d M2 = (R + P) / 2;
	vector3d D1 = cross(N, V);
	vector3d D2 = cross(N, W);
	double t;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	}
	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	}
	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	{
		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	}

	return M1 + t * D1;
}


std::string xUtilityFunctions::xstring(QString v)
{
	return v.toStdString();
}

void xUtilityFunctions::DeleteFilesInDirectory(std::string path)
{
	QString qpath = QString::fromStdString(path);
	QDir dir = QDir(qpath);
	dir.setNameFilters(QStringList() << "*.*");
	dir.setFilter(QDir::Files);
	foreach(QString dirFile, dir.entryList())
	{
		dir.remove(dirFile);
	}
}

void xUtilityFunctions::DeleteFileByEXT(std::string path, std::string ext)
{
	// QString dDir = model::path + model::name;
	QString qpath = QString::fromStdString(path);
	QString qext = QString::fromStdString(ext);
	QDir dir = QDir(qpath);
	QStringList delFileList;
	delFileList = dir.entryList(QStringList("*." + qext), QDir::Files | QDir::NoSymLinks);
	//qDebug() << "The number of *.bin file : " << delFileList.length();
	for (int i = 0; i < delFileList.length(); i++){
		QString deleteFilePath = qpath + "/" + delFileList[i];
		QFile::remove(deleteFilePath);
	}
}