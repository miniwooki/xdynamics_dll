#ifndef XUTILITYFUNCTIONS_H
#define XUTILITYFUNCTIONS_H

#include "xdynamics_decl.h"
#include <QtCore/QString>

#define kor(str) QString::fromLocal8Bit(str)

inline const char* ToChar(QString& qs) { return qs.toStdString().c_str(); }
inline const QString toWideCharToQString(const wchar_t* wc) { return QString::fromStdWString(std::wstring(wc)); }

class XDYNAMICS_API xUtilityFunctions
{
public:
	static std::string xstring(QString v);
	static std::string xstring(int v);
	static std::string xstring(unsigned int v);
	static std::string xstring(double v);
	static std::string xstring(std::wstring v);
	static std::string WideChar2String(const wchar_t* wc);
	static std::wstring Multibyte2WString(const char* c);
	static std::wstring GetDateTimeFormat(const char* format, int nseg);
	static int FindNumString(const string& s, const char* c);
	static void xsplit(const std::wstring& s, const char* c, int n, int* data);
	static void xsplit(const std::wstring& s, const char* c, vector2i& data);
	static void xsplit(const std::wstring& s, const char* c, int n, double* data);
	static void xsplit(const wchar_t* s, const char* c, int n, int* data);
	static void xsplit(const wchar_t* s, const char* c, int n, double* data);
	static void xsplit(const wchar_t* s, const char* c, int n, std::wstring* data);
	static unsigned int xsplitn(const wchar_t* s, const char* c);
	//static void Split2WString(const wchar_t* s, const wchar_t* c, int& n, QString* ws);
	static double AngleCorrection(double d, double th);
	static void DirectoryFileList(const wchar_t* _path);
	static void DeleteFileByEXT(std::string path, std::string s);
	static void DeleteFilesInDirectory(std::string path);
	static bool ExistFile(const wchar_t* n);
	static QString FileExtension(const wchar_t* f);
	static QString GetFileName(const wchar_t* p);
	static void CreateDirectory(const wchar_t* _path);
	static vector3d QuaternionRotation(vector4d& q, vector3d& v);
	static double SignedVolumeOfTriangle(vector3d& v1, vector3d& v2, vector3d& v3);
	static vector3d CenterOfTriangle(vector3d& P, vector3d& Q, vector3d& R);
};

#endif