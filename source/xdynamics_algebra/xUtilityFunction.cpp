//#include "stdafx.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "xdynamics_algebra/xAlgebraMath.h"
/*#include "boost/filesystem.hpp"*/
#include <ctime>
#include <sstream>
#include <filesystem>

//using namespace boost::filesystem;

//std::string xUtilityFunctions::xstring(int v)
//{
//	stringstream(s)
//	return QString("%1").arg(v).toStdString();
//}
//
//std::string xUtilityFunctions::xstring(unsigned int v)
//{
//	return QString("%1").arg(v).toStdString();
//}
//
//std::string xUtilityFunctions::xstring(double v)
//{
//	return QString("%1").arg(v).toStdString();
//}

void xUtilityFunctions::CreateDirectory(const char* _path)
{
	/*QString path = QString::fromStdString(_path);
	if (!QDir(path).exists())
		QDir().mkdir(path);
	*/
// 	try
// 	{
	
 	filesystem::path p(_path);
	if (!filesystem::exists(p))
		filesystem::create_directory(p);

// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::cout << ex.what() << std::endl;
// 		throw;
// 	}
}

vector3d xUtilityFunctions::QuaternionRotation(vector4d & q, vector3d & v)
{
	vector3d _q = new_vector3d(q.y, q.z, q.w);
	vector3d t = 2.0 * cross(_q, v);
	vector3d p_hat = v + q.x * t + cross(_q, t);
	return p_hat;
}

//std::string xUtilityFunctions::xstring(std::string v)
//{
//	return QString::fromStdString(v).toStdString();
//}

std::string xUtilityFunctions::GetFileName(const char* pn)
{
	std::string path = pn;// QString::fromStdString(pn);
	size_t begin = path.find_last_of('/') + 1;// finlastIndexOf('/');
	size_t end = path.find_last_of('.');// lastIndexOf('.');
	return path.substr(begin, end - begin);// path.mid(begin + 1, end - begin - 1);
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

std::string xUtilityFunctions::FileExtension(const char* f)
{
	//std::string path = f;
	////int begin = path.find_last_of('/') + 1;// finlastIndexOf('/');
	//int begin = path.find_last_of('.') + 1;// lastIndexOf('.');
	////int begin = path.lastIndexOf('.');
	//return path.substr(begin, path.size() - begin);
// 	
// 	try
// 	{
	filesystem::path p(f);
 	return p.extension().string();
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::cout << ex.what() << std::endl;
// 		throw;
// 	}
// 	return b;
}

bool xUtilityFunctions::ExistFile(const char* n)
{
	//QString path = QString::fromStdString(n);
	// QFile().exists(path);
// 	/bool b = false;
// 	try
// 	{
	filesystem::path p(n);
	return filesystem::exists(p);
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::cout << ex.what() << std::endl;
// 		throw;
// 	}
// 	return b;
}

void xUtilityFunctions::DirectoryFileList(const char* _path)
{
	//QString path = QString::fromStdString(_path);
	//QDir d(path);
	//QStringList ls = d.entryList();
	//foreach(QString s, ls)
	//{
	//	std::cout << "     " << s.toLocal8Bit().data() << std::endl;
	//}
// 	try
// 	{
	filesystem::path p(_path);
	if (filesystem::exists(p))
	{
		std::cout << std::endl << "File list of " << _path << std::endl;
		for (auto& p: std::filesystem::directory_iterator(p))
		{
			std::cout << "     " << p.path() << std::endl;
		}
	}
// 	}
// 	catch (boost::filesystem::filesystem_error& ex)
// 	{
// 		std::cout << ex.what() << std::endl;
// 		throw;
// 	}
}

//std::wstring xUtilityFunctions::Multibyte2WString(const char* c)
//{
//	int len = (int)strlen(c);
//	wchar_t* wcArr = NULL;// new char[len];// (char*)calloc(1, len);
//	len = MultiByteToWideChar(CP_ACP, 0, c, len, NULL, NULL);
//	wcArr = SysAllocStringLen(NULL, len);
//	MultiByteToWideChar(CP_ACP, 0, c, len, wcArr, len);
//	wstring cstr = wcArr;
//	SysFreeString(wcArr);// [] wcArr;
//	return cstr;
//}
//
 std::string xUtilityFunctions::WideChar2String(const wchar_t* wc)
 {
 	int wlen;
 	char* carr;
 
 	wlen = (int)wcslen(wc);
 	carr = (char*)calloc(1, wlen * 2 + 1);
 	WideCharToMultiByte(CP_ACP, 0, wc, wlen, carr, wlen * 2, 0, 0);
 	string cstr = carr;
 	delete carr;
 	return cstr;
 }

std::string xUtilityFunctions::GetDateTimeFormat(const char* format, int nseg)
{
	time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);
	rawtime += nseg;
	localtime_s(&timeinfo, &rawtime);
	char bufftime[64];
	strftime(bufftime, 64, format, &timeinfo);
	return bufftime;
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

//void xUtilityFunctions::xsplit(const std::string& s, const char* c, int n, int* data)
//{
//	QString d = QString::fromStdString(s);
//	QStringList ds = d.split(c);
//	for (int i = 0; i < n; i++)
//		data[i] = ds.at(i).toInt();
//}
//
//void xUtilityFunctions::xsplit(const std::string& s, const char* c, vector2i& data)
//{
//	QString d = QString::fromStdString(s);
//	QStringList ds = d.split(c);
//	data.x = ds.at(0).toDouble();
//	data.y = ds.at(1).toDouble();
//}
//
//void xUtilityFunctions::xsplit(const std::string& s, const char* c, int n, double* data)
//{
//	QString d = QString::fromStdString(s);
//	QStringList ds = d.split(c);
//	for (int i = 0; i < n; i++)
//		data[i] = ds.at(i).toDouble();
//}
//
//bool xUtilityFunctions::xsplit(const char* wc, const char* c, int n, int* data)
//{
//	QString d = toWideCharToQString(wc);
//	QStringList ds = d.split(c);
//	if (ds.size() != n)
//		return false;
//	for (int i = 0; i < n; i++)
//		data[i] = ds.at(i).toInt();
//	return true;
//// 	string s = WideChar2String(wc);
//// 	basic_string<char>::size_type start = 0, end;
//// 	static const basic_string<char>::size_type npos = -1;
//// 	int ns = 0;
//// 	int len = (int)strlen(c);
//// 	while (1)
//// 	{
//// 		end = s.find(c, start);
//// 		string cs = s.substr(start, end - start);
//// 		data[ns] = atoi(cs.c_str());
//// 		start = end + len;
//// 		ns++;
//// 		if (end == npos)
//// 			break;
//// 	}
//}
//
//void xUtilityFunctions::xsplit(const char* wc, const char* c, int n, double* data)
//{
//	QString d = toWideCharToQString(wc);
//	QStringList ds = d.split(c);
//	for (int i = 0; i < n; i++)
//		data[i] = ds.at(i).toDouble();
//// 	string s = WideChar2String(wc);
//// 	basic_string<char>::size_type start = 0, end;
//// 	static const basic_string<char>::size_type npos = -1;
//// 	int ns = 0;
//// 	int len = (int)strlen(c);
//// 	while (1)
//// 	{
//// 		end = s.find(c, start);
//// 		string cs = s.substr(start, end - start);
//// 		data[ns] = atof(cs.c_str());
//// 		start = end + len;
//// 		ns++;
//// 		if (end == npos)
//// 			break;
//// 	}
//}
//
//unsigned int xUtilityFunctions::xsplitn(const char* s, const char* c)
//{
//	QString d = toWideCharToQString(s);
//	return d.split(c).size();
//}
//
//void xUtilityFunctions::xsplit(const char* wc, const char* c, int n, std::string* data)
//{
//	QString d = toWideCharToQString(wc);
//	QStringList ds = d.split(c);
//	for (int i = 0; i < n; i++)
//		data[i] = ds.at(i).toStdString();
//	// 	string s = WideChar2String(wc);
//	// 	basic_string<char>::size_type start = 0, end;
//	// 	static const basic_string<char>::size_type npos = -1;
//	// 	int ns = 0;
//	// 	int len = (int)strlen(c);
//	// 	while (1)
//	// 	{
//	// 		end = s.find(c, start);
//	// 		string cs = s.substr(start, end - start);
//	// 		data[ns] = atof(cs.c_str());
//	// 		start = end + len;
//	// 		ns++;
//	// 		if (end == npos)
//	// 			break;
//	// 	}
//}

// void xUtilityFunctions::Split2WString(const char* wc, const char* c, int& n, wstring* data)
// {
// 	n = 0;
// 	wstring s = wc;
// 	basic_string<char>::size_type start = 0, end;
// 	static const basic_string<char>::size_type npos = -1;
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
	double t=0.0;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
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

vector4d xUtilityFunctions::FitSphereToTriangle(vector3d& P, vector3d& Q, vector3d& R, double ft)
{
	vector3d V = Q - P;
	vector3d W = R - P;
	vector3d N = cross(V, W);
	N = N / length(N);// .length();
	vector3d M1 = (Q + P) / 2;
	vector3d M2 = (R + P) / 2;
	vector3d D1 = cross(N, V);
	vector3d D2 = cross(N, W);
	double t=0.0;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
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
	vector3d Ctri = M1 + t * D1;
	vector3d Csph = new_vector3d(0.0, 0.0, 0.0);
	double fc = 0;
	double tol = 1e-4;
	double r = length(P - Ctri);
	while (abs(fc - ft) > tol)
	{
		double d = ft * r;
		double p = d / length(N);
		Csph = Ctri - p * N;
		r = length(P - Csph);
		fc = d / r;
	}
	return new_vector4d(Csph.x, Csph.y, Csph.z, r);
}

double xUtilityFunctions::FitClusterRadius(vector4d * cpos, unsigned int n)
{
	unsigned int cnt = 0;
	double maximum = -FLT_MAX;
	unsigned int m_i = 0;
	unsigned int m_j = 0;
	vector4d cpi, cpj;
	vector3d ci, cj;
	while (cnt + 1 < n)
	{
		cpi = cpos[cnt];
		ci = new_vector3d(cpi.x, cpi.y, cpi.z);
		for (unsigned int i = cnt + 1; i < n; i++)
		{
			cpj = cpos[i];
			cj = new_vector3d(cpj.x, cpj.y, cpj.z);
			double len_c = length(cj - ci);
			double dij = length((cj - ci) + cpj.w * ((cj - ci) / len_c) + cpi.w * ((cj - ci) / len_c));
			if (dij > maximum)
			{
				maximum = dij;
				m_i = cnt;
				m_j = i;
			}
		}
		cnt++;
	}
	cpi = cpos[m_i];
	cpj = cpos[m_j];
	ci = new_vector3d(cpi.x, cpi.y, cpi.z);
	cj = new_vector3d(cpj.x, cpj.y, cpj.z);
	vector3d Pi = ci + cpi.w * ((ci - cj) / length(ci - cj));
	vector3d Pj = cj + cpj.w * ((cj - ci) / length(cj - ci));
	vector3d Ca = 0.5 * (Pi + Pj);
	double Ra = 0.5 * length(Pi - Pj);
	vector3d Cb = new_vector3d(0, 0, 0);
	for (unsigned int i = 0; i < n; i++)
	{
		if (i == m_i || i == m_j)
			continue;
		vector4d CPo = cpos[i];
		vector3d Co = new_vector3d(CPo.x, CPo.y, CPo.z);
		vector3d Pa = Ca + (Co - Ca) + CPo.w * ((Co - Ca) / length(Co - Ca)); 
		double len_a = length(Pa - Ca);
		if (len_a > Ra)
		{
			Cb = Pa - Ra * ((Ca - Pa) / length(Ca - Pa));
			Ra = len_a;
		}
	}
	return Ra;
}

double xUtilityFunctions::CriticalTimeStep(double min_rad, double rho, double E, double p)
{
	double dt_raleigh = M_PI * min_rad * sqrt(rho * 2.0 * (1 + p) / E) / (0.1631 * (p + 0.8766));
	double dt_hertz = 2.87 * pow(pow(rho * (4.0 / 3.0) * M_PI * pow(min_rad, 3.0), 2.0) / (min_rad * E * E * 1.0), 0.2);
	double dt_cundall = 0.2 * M_PI * sqrt(rho * (4.0 / 3.0) * M_PI * min_rad * min_rad * 3.0 * (1.0 + 2.0 * p) / (E * 0.01));
	double min_dt = dt_raleigh < dt_hertz ? (dt_raleigh < dt_cundall ? dt_raleigh : dt_cundall) : (dt_hertz < dt_cundall ? dt_hertz : dt_cundall);
	return 0.2 * min_dt;
}

void xUtilityFunctions::DeleteFilesInDirectory(std::string path)
{
	//QString dDir = QString::fromStdString(path) + "/";
	//QDir dir = QDir(dDir);
	//QStringList delFileList;
	//delFileList = dir.entryList(QStringList("*.*"), QDir::Files | QDir::NoSymLinks);
	////qDebug() << "The number of *.bin file : " << delFileList.length();
	//for (int i = 0; i < delFileList.length(); i++){
	//	QString deleteFilePath = dDir + delFileList[i];
	//	QFile::remove(deleteFilePath);
	//}
// 	qDebug() << "Complete delete.";
// 	QString qpath = QString::fromStdString(path);
// 	QDir dir = QDir(qpath);
//   	dir.setNameFilters(QStringList() << "*.*");
//   	dir.setFilter(QDir::Files);
// 	foreach(QString dirFile, dir.entryList())
// 	{
// 		QFile::remove(dirFile);
// 		//dir.remove(dirFile);
// 	}
}

void xUtilityFunctions::DeleteFileByEXT(std::string path, std::string ext)
{
	// QString dDir = model::path + model::name;
	//QString qpath = QString::fromStdString(path);
	//QString qext = QString::fromStdString(ext);
	//QDir dir = QDir(qpath);
	//QStringList delFileList;
	//delFileList = dir.entryList(QStringList("*." + qext), QDir::Files | QDir::NoSymLinks);
	////qDebug() << "The number of *.bin file : " << delFileList.length();
	//for (int i = 0; i < delFileList.length(); i++){
	//	QString deleteFilePath = qpath + "/" + delFileList[i];
	//	QFile::remove(deleteFilePath);
	//}
}

double xUtilityFunctions::RelativeAngle(int udrl, double theta, unsigned int& n_rev, vector3d& gi, vector3d& fi, vector3d& fj)
{
	double df = dot(fi, fj);
	double dg = dot(gi, fj);
	double a = acos(dot(fi, fj));
	double b = asin(dg);
	double a_deg = a * 180 / M_PI;
	double b_deg = b * 180 / M_PI;
	double stheta = 0.0;
	double p_theta = theta;
	if ((df <= 0.2 && df > -0.8) && dg > 0) { udrl = UP_RIGHT;  stheta = acos(df); }
	else if ((df < -0.8 && df >= -1.1 && dg > 0) || (df > -1.1 && df <= -0.2 && dg < 0)) { udrl = UP_LEFT; stheta = M_PI - asin(dg); }
	else if ((df > -0.2 && df <= 0.8) && dg < 0) { udrl = DOWN_LEFT; stheta = 2.0 * M_PI - acos(df); }
	else if ((df > 0.8 && df < 1.1 && dg < 0) || (df <= 1.1 && df > 0.2 && dg > 0)) { udrl = DOWN_RIGHT; stheta = 2.0 * M_PI + asin(dg); }
	if (p_theta >= 2.0 * M_PI && stheta < 2.0 * M_PI)
		n_rev--;
	if (p_theta > M_PI && p_theta < 2.0 * M_PI && stheta >= 2.0 * M_PI)
		n_rev++;

	if (stheta >= 2.0 * M_PI)
		stheta -= 2.0 * M_PI;
	//std::cout << "stheta : " << stheta << std::endl;
	return stheta;// xUtilityFunctions::AngleCorrection(prad, stheta);
}

//double xUtilityFunctions::DerivativeRelativeAngle(
//	int udrl, double theta, unsigned int& n_rev, 
//	vector3d& gi, vector3d& fi, vector3d& fj, 
//	vector3d& dgi, vector3d& dfi, vector3d& dfj)
//{
//	double df = dot(fj, dfi) + dot(fi, dfj);
//	double dg = dot(fj, dgi) + dot(gi, dfj);// dot(gi, fj);
//	double a = -asin(df);
//	double b = acos(dg);
//	double a_deg = a * 180 / M_PI;
//	double b_deg = b * 180 / M_PI;
//	double stheta = 0.0;
//	double p_theta = theta;
//	if ((df <= 0.2 && df > -0.8) && dg > 0) { udrl = UP_RIGHT;  stheta = acos(df); }
//	else if ((df < -0.8 && df >= -1.1 && dg > 0) || (df > -1.1 && df <= -0.2 && dg < 0)) { udrl = UP_LEFT; stheta = M_PI - asin(dg); }
//	else if ((df > -0.2 && df <= 0.8) && dg < 0) { udrl = DOWN_LEFT; stheta = 2.0 * M_PI - acos(df); }
//	else if ((df > 0.8 && df < 1.1 && dg < 0) || (df <= 1.1 && df > 0.2 && dg > 0)) { udrl = DOWN_RIGHT; stheta = 2.0 * M_PI + asin(dg); }
//	if (p_theta >= 2.0 * M_PI && stheta < 2.0 * M_PI)
//		n_rev--;
//	if (p_theta > M_PI && p_theta < 2.0 * M_PI && stheta >= 2.0 * M_PI)
//		n_rev++;
//
//	if (stheta >= 2.0 * M_PI)
//		stheta -= 2.0 * M_PI;
//	//std::cout << "stheta : " << stheta << std::endl;
//	return stheta;// xUtilityFunctions::AngleCorrection(prad, stheta);
//}