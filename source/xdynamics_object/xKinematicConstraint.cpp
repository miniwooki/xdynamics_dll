#include "xdynamics_object/xKinematicConstraint.h"
#include <sstream>

xKinematicConstraint::xKinematicConstraint()
	: i(-2)
	, j(-2)
	, i_ptr(NULL)
	, j_ptr(NULL)
	, nConst(0)
	, nr_part(0)
	//, kcrs(NULL)
{
	//name = { 0, };
	//memset(name, 0, sizeof(*this));
}

xKinematicConstraint::xKinematicConstraint(std::string _name, cType _type, std::string _i, std::string _j)
	: type(_type)
	, i(-2)
	, j(-2)
	, i_ptr(NULL)
	, j_ptr(NULL)
	, nConst(0)
	, nr_part(0)
	//, kcrs(NULL)
{
	//name = { 0, };
	//memset(name, 0, sizeof(*this) - sizeof(name));
	int mem_sz = sizeof(vector3d) * 6;
	memset(&location.x, 0, mem_sz);
	name = _name;// wsprintfW(name, TEXT("%s"), _name);
	switch (_type)
	{
	case FIXED: nConst = 6; break;
	case SPHERICAL: nConst = 3; break;
	case REVOLUTE: nConst = 5; break;
	case TRANSLATIONAL: nConst = 5; break;
	case UNIVERSAL: nConst = 4; break;
	}
	base = _i;
	action = _j;
	//djaco.alloc(49 * 4);
}

xKinematicConstraint::~xKinematicConstraint()
{
	//if (kcrs) delete[] kcrs; kcrs = NULL;
}

std::string xKinematicConstraint::Name()
{
	return name.toStdString();
}

xKinematicConstraint::cType xKinematicConstraint::Type()
{
	return type;
}

unsigned int xKinematicConstraint::NumConst()
{
	return nConst;
}

int xKinematicConstraint::IndexBaseBody()
{
	return i;
}

int xKinematicConstraint::IndexActionBody()
{
	return j;
}

std::string xKinematicConstraint::BaseBodyName()
{
	return base.toStdString();
}

std::string xKinematicConstraint::ActionBodyName()
{
	return action.toStdString();
}

xPointMass* xKinematicConstraint::BaseBody()
{
	return i_ptr;
}

xPointMass* xKinematicConstraint::ActionBody()
{
	return j_ptr;
}

void xKinematicConstraint::setBaseBodyIndex(int _i)
{
	i = _i;
}

void xKinematicConstraint::setActionBodyIndex(int _j)
{
	j = _j;
}

void xKinematicConstraint::AllocResultMemory(unsigned int _s)
{
	//if (kcrs.size())
	//{
	//	kcrs.clear();/// delete[] kcrs;
	//	nr_part = 0;
	//	//kcrs = NULL;
	//}
	//kcrs = new kinematicConstraint_result[_s];
}

void xKinematicConstraint::setLocation(double x, double y, double z)
{
	location.x = x;
	location.y = y;
	location.z = z;
}

void xKinematicConstraint::SetupDataFromStructure(xPointMass* base, xPointMass* action, xJointData& d)
{
	i_ptr = base;
	j_ptr = action;
	location = new_vector3d(d.lx, d.ly, d.lz);
	spi = i_ptr->toLocal(location - i_ptr->Position());
	fi = new_vector3d(d.fix, d.fiy, d.fiz);
	gi = new_vector3d(d.gix, d.giy, d.giz);
	spj = j_ptr->toLocal(location - j_ptr->Position());
	fj = new_vector3d(d.fjx, d.fjy, d.fjz);
	gj = new_vector3d(d.gjx, d.gjy, d.gjz);
	hi = cross(fi, gi);
	hj = cross(fj, gj);
}

//QVector<xKinematicConstraint::kinematicConstraint_result>* xKinematicConstraint::XKinematicConstraintResultPointer()
//{
//	return &kcrs;
//}

void xKinematicConstraint::ImportResults(std::string f)
{
	//std::fstream fs;
	//fs.open(f, ios_base::in | ios_base::binary);
	//char t = 'e';
	//int identifier = 0;
	//fs.read((char*)&identifier, sizeof(int));
	//fs.read(&t, sizeof(char));
	//fs.read((char*)&nr_part, sizeof(unsigned int));
	//for (unsigned int i = 0; i < nr_part; i++)
	//{
	//	kinematicConstraint_result kr = { 0, };
	//	fs.read((char*)&kr, sizeof(kinematicConstraint_result));
	//	kcrs.push_back(kr);
	//}
	////fs.read((char*)kcrs.data(), sizeof(kinematicConstraint_result) * nr_part);
	//fs.close();
}

void xKinematicConstraint::ExportResults(std::fstream& of)
{
	//std::ofstream ofs;
	//std::string _path;
	//stringstream ss(_path);
	//ss << (xModel::path + xModel::name + "/").toStdString() + name.toStdString() + ".bkc";
	////QString _path = xModel::path + xModel::name + "/" + name + ".bkc";
	////QString _path = QString(xModel::path) + QString(xModel::name) + "/" + QString(name) + ".bkc";
	//ofs.open(_path, ios::binary | ios::out);
	//char t = 'k';
	//int identifier = RESULT_FILE_IDENTIFIER;
	//ofs.write((char*)&identifier, sizeof(int));
	//ofs.write(&t, sizeof(char));
	//ofs.write((char*)&nr_part, sizeof(unsigned int));
	//ofs.write((char*)kcrs.data(), sizeof(kinematicConstraint_result) * nr_part);
	//// 	ofs << "time " << "px " << "py " << "pz " << "vx " << "vy " << "vz " << "ax " << "ay " << "az " 
	//// 		<< "avx " << "avy " << "avz " << "aax " << "aay " << "aaz " 
	//// 		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz " 
	//// 		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
	//// 		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
	//ofs.close();
	//xLog::log("Exported : " + _path);
	//of << _path << endl;
	//std::cout << "Exported : " << _path.text() << std::endl;
}

matrix34d xKinematicConstraint::spherical_constraintJacobian_e(euler_parameters& e, vector3d& s)
{
	return BMatrix(e, s);
}

void xKinematicConstraint::spherical_differentialJacobian(xMatrixD& lhs, vector3d& L)
{
	unsigned int idx = (i - 1) * xModel::OneDOF();
	unsigned int jdx = (j - 1) * xModel::OneDOF();
	
	if (i) lhs.plus(idx + 3, idx + 3, DMatrix(-spi, L));
	if (j) lhs.plus(jdx + 3, jdx + 3, DMatrix(spj, L));
}

vector3d xKinematicConstraint::spherical_constraintEquation(
	vector3d& ri, vector3d& rj, vector3d& si, vector3d& sj)
{
// 	matrix33d Ai = GlobalTransformationMatrix(ei);
// 	matrix33d Aj = GlobalTransformationMatrix(ej);
	return (rj + sj) - (ri + si);
}

vector3d xKinematicConstraint::spherical_gamma(euler_parameters dei, euler_parameters dej, vector3d si, vector3d sj) 
{
	vector3d gamma_SPH;
	gamma_SPH = -BMatrix(dei, si)*dei + BMatrix(dej, sj)*dej;
	return gamma_SPH;
}

vector4d xKinematicConstraint::dot_1_constraintJacobian(vector3d& si, vector3d& sj, euler_parameters& e)
{
	return si * BMatrix(e, sj);//lhs.insert(sr + 3, sc + 3, (A * hj) * BMatrix(e, fi));
}

double xKinematicConstraint::dot_1_constraintEquation(vector3d& s1, vector3d& s2)
{
	return dot(s1, s2);
}

void xKinematicConstraint::dot_1_differentialJacobian(
	xMatrixD& lhs, vector3d& si, vector3d& sj, euler_parameters& ei, euler_parameters& ej, double L)
{
	unsigned int idx = (i - 1) * xModel::OneDOF();
	unsigned int jdx = (j - 1) * xModel::OneDOF();
	//matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	//matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix44d Di = L * DMatrix(si, j_ptr->toGlobal(sj));// +lm[4] * DMatrix(gi, Aj * hj);
	matrix44d Dj = L * DMatrix(sj, i_ptr->toGlobal(si));// Ai * si);// +lm[3] * DMatrix(hj, Ai * fi) + lm[4] * DMatrix(hj, Ai * gi);

	if (i) lhs.plus(idx + 3, idx + 3, Di);
	if (j) lhs.plus(jdx + 3, jdx + 3, Dj);

	Di = L * BMatrix(ei, si) * BMatrix(ej, sj);// +lm[4] * BMatrix(ei, gi) * BMatrix(ej, hj);
	Dj = L * BMatrix(ej, sj) * BMatrix(ei, si);// +lm[4] * BMatrix(ej, hj) * BMatrix(ei, gi);

	if (i && j)
	{
		lhs.plus(idx + 3, jdx + 3, Di);
		lhs.plus(jdx + 3, idx + 3, Dj);
	}
}

double xKinematicConstraint::dot_1_gamma(euler_parameters ei, euler_parameters ej, vector3d ai, vector3d aj, euler_parameters dei, euler_parameters dej)
{
	double gamma_DOT1 = 0;
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);

	gamma_DOT1 = dot(aj, Transpose(Aj) * (BMatrix(dei, ai)*dei)) + 2.0 * dot(dei, (BMatrix(ei, ai) * BMatrix(ej, aj)) * dej) + dot(ai, Transpose(Ai)*BMatrix(dej, aj)*dej);
	return gamma_DOT1;
}

vector4d xKinematicConstraint::dot_2_constraintJacobian(vector3d& s1_global, vector3d& s2_local, euler_parameters& e)
{
	return s1_global * BMatrix(e, s2_local);
}

void xKinematicConstraint::dot_2_differentialJacobian(
	xMatrixD& lhs, vector3d& a_local, vector3d& si_local, vector3d& sj_local, vector3d& dij, euler_parameters& ei, euler_parameters& ej, double L)
{
	unsigned int idx = (i - 1) * xModel::OneDOF();
	unsigned int jdx = (j - 1) * xModel::OneDOF();
//	matrix33d Ai = GlobalTransformationMatrix(ei);
	//matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix34d Bpa = BMatrix(ei, a_local);
	matrix34d Bps = BMatrix(ej, sj_local);
	matrix44d Das = DMatrix(a_local, dij + i_ptr->toGlobal(si_local));// Ai * si_local);
	matrix44d Dsa = DMatrix(sj_local, i_ptr->toGlobal(a_local));// Ai * a_local);
	if (i)
	{
		lhs.plus(idx + 0, idx + 3, -Bpa);
		lhs.plus(idx + 3, idx + 0, -Bpa, true);
		lhs.plus(idx + 3, idx + 3, Das);
	}
	if (j)
	{
		lhs.plus(jdx + 3, jdx + 3, Dsa);
	}
	if (i && j)
	{
		lhs.plus(idx + 3, jdx + 0, Bpa, true);
		lhs.plus(idx + 3, jdx + 3, Bpa * Bps);
		lhs.plus(jdx + 0, idx + 3, Bpa);
		lhs.plus(jdx + 3, idx + 3, Bps * Bpa);
	}
}

double xKinematicConstraint::dot_2_constraintEquation(vector3d& s1_global, vector3d& s2_global, vector3d& s1_local, vector3d& s2_local)
{
	return dot(s1_global, s2_global) - dot(s1_local, s2_local);
}

double xKinematicConstraint::dot_2_gamma(euler_parameters ei, euler_parameters ej, vector3d aj, vector3d si, vector3d sj, vector3d dij, vector3d drj, vector3d dri, euler_parameters dei, euler_parameters dej)
{
	double Gamma_Dot2;
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	Gamma_Dot2 = 2 * dot(dej, BMatrix(ej, sj)*(BMatrix(ej, aj)*dej)) + dot(dij, BMatrix(dej, aj)*dej)
		+ dot(aj, Aj*(BMatrix(dej, sj)*dej)) - dot(aj, Aj*(BMatrix(dei, si)*dei)) + 2 * dot(drj, BMatrix(ej, aj)*dej)
		- 2 * dot(dri, BMatrix(ej, aj)*dej) - 2 * dot(dei, BMatrix(ei, si)*(BMatrix(ej, aj)*dej));
	return -Gamma_Dot2;
}

double xKinematicConstraint::relative_rotation_constraintGamma(double theta, euler_parameters& ei, euler_parameters& ej, euler_parameters& dei, euler_parameters& dej, vector3d& global_fi, vector3d& global_gi, vector3d& global_fj)
{
	double v = 0.0;
	vector3d B0dpj = BMatrix(dej, fj) * dej;
	vector3d B1dpj = BMatrix(ej, fj) * dej;
	v = -cos(theta) * (dot(global_fj, BMatrix(dei, gi) * dei) + dot(global_gi, B0dpj) + 2.0 * dot(dei, BMatrix(ei, gi) * B1dpj))
		+sin(theta) * (dot(global_fj, BMatrix(dei, fi) * dei) + dot(global_fi, B0dpj) + 2.0 * dot(dei, BMatrix(ei, fi) * B1dpj));
	return -v;
}

double xKinematicConstraint::relative_translation_constraintGamma(vector3d& dij, vector3d& global_ai, vector3d& local_ai, vector3d& vi, vector3d& vj, euler_parameters& ei, euler_parameters& ej, euler_parameters& dei, euler_parameters& dej)
{
	double v = 0.0;
	vector3d B0 = BMatrix(ei, local_ai) * dei;
	v = dot(vi, B0) - dot(dij, BMatrix(dei, local_ai) * dei) + dot(global_ai, BMatrix(dei, spi) * dei) + dot(dei, BMatrix(ei, spi) * B0)
		- dot(vj, B0) - dot(global_ai, BMatrix(dej, spj) * dej) - dot(dej, BMatrix(ej, spj) * B0)
		- dot(dei, BMatrix(ei, local_ai) * (vj + BMatrix(ej, spj) * dej - vi - B0));
	return -v;
}

vector4d xKinematicConstraint::relative_rotation_constraintJacobian_e_i(
	double theta, euler_parameters& ei, euler_parameters& ej, vector3d& global_fj)
{
	vector4d D1 = global_fj * (cos(theta) * BMatrix(ei, gi) - sin(theta) * BMatrix(ei, fi));// transpose(fj, cos(theta) * B(im->getEP(), kconst->g_i()));
	return D1;
}

vector4d xKinematicConstraint::relative_rotation_constraintJacobian_e_j(
	double theta, euler_parameters& ei, euler_parameters& ej, vector3d& global_fi, vector3d& global_gi)
{
	vector4d D2 = (cos(theta) * global_gi - sin(theta) * global_fi) * BMatrix(ej, fj);
	return D2;
}