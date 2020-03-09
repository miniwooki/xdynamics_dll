#ifndef XUTILITYFUNCTIONS_H
#define XUTILITYFUNCTIONS_H

#include "xdynamics_decl.h"
#include <vector>

//inline const char* ToChar(QString& qs) { return qs.toStdString().c_str(); }
//inline const QString toWideCharToQString(const char* wc) { return QString::fromStdString(std::string(wc)); }

class XDYNAMICS_API xUtilityFunctions
{
public:
	static std::string WideChar2String(const wchar_t* wc);
	static std::string GetDateTimeFormat(const char* format, int nseg);
	static int FindNumString(const string& s, const char* c);
	static double AngleCorrection(double d, double th);
	static void DirectoryFileList(const char* _path);
	static void DeleteFileByEXT(std::string path, std::string s);
	static void DeleteFilesInDirectory(std::string path);
	static bool ExistFile(const char* n);
	static std::string FileExtension(const char* f);
	static std::string GetFileName(const char* p);
	static void CreateDirectory(const char* _path);
	static vector3d QuaternionRotation(vector4d& q, vector3d& v);
	static double SignedVolumeOfTriangle(vector3d& v1, vector3d& v2, vector3d& v3);
	static vector3d CenterOfTriangle(vector3d& P, vector3d& Q, vector3d& R);
	static vector4d FitSphereToTriangle(vector3d& P, vector3d& Q, vector3d& R, double ft);
	static double FitClusterRadius(vector4d *cpos, unsigned int n, double specificRadius);
	static double CriticalTimeStep(double min_rad, double rho, double E, double p);
	static double RelativeAngle(int udrl, double theta, int& n_rev, vector3d& gi, vector3d& fi, vector3d& fj, bool &isSin);
	static std::vector<__int64> RandomDistribution(__int64 s, __int64 e, bool useSeed);
};

#endif