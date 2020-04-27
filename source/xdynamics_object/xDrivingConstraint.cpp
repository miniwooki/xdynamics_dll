#include "xdynamics_object/xDrivingConstraint.h"
#include "xdynamics_object/xKinematicConstraint.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include <sstream>

xDrivingConstraint::xDrivingConstraint()
	: type(ROTATION_DRIVING)
	, plus_time(0.0)
	, start_time(0.0)
	, end_time(FLT_MAX)
	, is_fix_after_end(false)
	, init_v(0.0)
	, cons_v(0.0)
	, theta(0.0)
	, kconst(NULL)
	, n_rev(0)
	, dn_rev(0)
	, srow(0)
	, udrl(0)
	, d_udrl(0)
{

}

xDrivingConstraint::xDrivingConstraint(std::string _name, xKinematicConstraint* _kc)
	: type(ROTATION_DRIVING)
	, plus_time(0.0)
	, start_time(0.0)
	, end_time(FLT_MAX)
	, is_fix_after_end(false)
	, init_v(0.0)
	, cons_v(0.0)
	, theta(0.0)
	, kconst(_kc)
	, n_rev(0)
	, dn_rev(0)
	, srow(0)
	, udrl(0)
	, d_udrl(0)
{
	if (kconst->Type() == xKinematicConstraint::REVOLUTE)
		type = ROTATION_DRIVING;
	else if (kconst->Type() == xKinematicConstraint::TRANSLATIONAL)
		type = TRANSLATION_DRIVING;

	name = _name;// wsprintfW(name, TEXT("%s"), _name);
}

xDrivingConstraint::~xDrivingConstraint()
{

}

unsigned int xDrivingConstraint::RevolutionCount()
{
	return n_rev;
}

unsigned int xDrivingConstraint::DerivativeRevolutionCount()
{
	return dn_rev;
}

double xDrivingConstraint::RotationAngle()
{
	return theta;
}

double xDrivingConstraint::EndTime()
{
	return end_time;
}

void xDrivingConstraint::setRevolutionCount(unsigned int _n_rev)
{
	n_rev = _n_rev;
}

void xDrivingConstraint::setDerivativeRevolutionCount(unsigned int _dn_rev)
{
	dn_rev = _dn_rev;
}

void xDrivingConstraint::setRotationAngle(double _theta)
{
	theta = _theta;
}

void xDrivingConstraint::define(xVectorD& q)
{
	unsigned int i = kconst->IndexBaseBody() * xModel::OneDOF();
	unsigned int j = kconst->IndexActionBody() * xModel::OneDOF();
	euler_parameters ei = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ej = new_euler_parameters(q(j + 3), q(j + 4), q(j + 5), q(j + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	if (type == ROTATION_DRIVING)
	{
		vector3d fi = Ai * kconst->Fi();// im->toGlobal(kconst->f_i());
		vector3d fj = Aj * kconst->Fj();
		init_v = acos(dot(fi, fj));

	}
	else if (type == TRANSLATION_DRIVING)
	{
		vector3d ri = new_vector3d(q(i + 0), q(i + 1), q(i + 2));
		vector3d rj = new_vector3d(q(j + 0), q(j + 1), q(j + 2));
		vector3d dij = (rj + Aj * kconst->Spj()) - (ri + Ai * kconst->Spi());
		init_v = length(dij);
	}
}

std::string xDrivingConstraint::Name()
{
	return name.toStdString();
}

int xDrivingConstraint::get_driving_type()
{
	return type;
}

void xDrivingConstraint::setStartTime(double st)
{
	start_time = st;
}

void xDrivingConstraint::setEndTime(double et)
{
	end_time = et;
}

void xDrivingConstraint::setConstantVelocity(double cv)
{
	cons_v = cv;
}

void xDrivingConstraint::setFixAfterDrivingEnd(bool b)
{
	is_fix_after_end = b;
}

void xDrivingConstraint::ImportResults(std::string f)
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
	//	xKinematicConstraint::kinematicConstraint_result kr = { 0, };
	//	fs.read((char*)&kr, sizeof(xKinematicConstraint::kinematicConstraint_result));
	//	kcrs.push_back(kr);
	//}
	////fs.read((char*)kcrs.data(), sizeof(kinematicConstraint_result) * nr_part);
	//fs.close();
}

void xDrivingConstraint::ExportResults(std::fstream & of)
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
	//ofs.write((char*)kcrs.data(), sizeof(xKinematicConstraint::kinematicConstraint_result) * nr_part);
	//// 	ofs << "time " << "px " << "py " << "pz " << "vx " << "vy " << "vz " << "ax " << "ay " << "az " 
	//// 		<< "avx " << "avy " << "avz "x << "aax " << "aay " << "aaz " 
	//// 		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz " 
	//// 		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
	//// 		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
	//ofs.close();
	//xLog::log("Exported : " + _path);
	//of << _path << endl;
}

void xDrivingConstraint::ConstraintGamma(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct, double mul)
{
	unsigned int i = kconst->IndexBaseBody() * xModel::OneDOF();
	unsigned int j = kconst->IndexActionBody() * xModel::OneDOF();
	vector3d ri = new_vector3d(q(i + 0), q(i + 1), q(i + 2));
	vector3d rj = new_vector3d(q(j + 0), q(j + 1), q(j + 2));
	vector3d vi = new_vector3d(qd(i + 0), qd(i + 1), qd(i + 2));
	vector3d vj = new_vector3d(qd(j + 0), qd(j + 1), qd(j + 2));
	euler_parameters ei = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ej = new_euler_parameters(q(j + 3), q(j + 4), q(j + 5), q(j + 6));
	euler_parameters dei = new_euler_parameters(qd(i + 3), qd(i + 4), qd(i + 5), qd(i + 6));
	euler_parameters dej = new_euler_parameters(qd(j + 3), qd(j + 4), qd(j + 5), qd(j + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	vector3d fi = Ai * kconst->Fi();
	vector3d fj = Aj * kconst->Fj();
	vector3d gi = Ai * kconst->Gi();
	if (type == ROTATION_DRIVING)
	{
		double v = kconst->relative_rotation_constraintGamma(theta, ei, ej, dei, dej, fi, gi, fj);
		rhs(sr) = mul * v;
	}
	else if (type == TRANSLATION_DRIVING)
	{
		vector3d dist = (rj + Aj * kconst->Spj()) - (ri + Ai * kconst->Spi());// kconst->CurrentDistance();VEC3D fdij = kconst->CurrentDistance();
		vector3d hi = Ai * kconst->Hi();// im->toGlobal(kconst->h_i());
		double v = kconst->relative_translation_constraintGamma(dist, hi, kconst->Hi(), vi, vj, ei, ej, dei, dej);
		rhs(sr) = mul * v;
	}
}

void xDrivingConstraint::ConstraintEquation(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct, double mul)
{
	double v = 0.0;
 	unsigned int i = kconst->IndexBaseBody() * xModel::OneDOF();
 	unsigned int j = kconst->IndexActionBody() * xModel::OneDOF();
	vector3d ri = new_vector3d(q(i + 0), q(i + 1), q(i + 2));
	vector3d rj = new_vector3d(q(j + 0), q(j + 1), q(j + 2));
	euler_parameters ei = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ej = new_euler_parameters(q(j + 3), q(j + 4), q(j + 5), q(j + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	if (type == TRANSLATION_DRIVING)
	{
		vector3d dist = (rj + Aj * kconst->Spj()) - (ri + Ai * kconst->Spi());// kconst->CurrentDistance();
		vector3d hi = Ai * kconst->Hi();// kconst->iMass()->toGlobal(kconst->h_i());
		double pct = ct + plus_time;
		if (start_time > pct || end_time < pct)
			if (is_fix_after_end)
				v = dot(hi, dist) - (init_v + end_time * ct);
			else
				v = dot(hi, dist) - (init_v + 0.0 * ct);
		else
			v = dot(hi, dist) - (init_v + cons_v * (ct - start_time));
		rhs(sr) = mul * v;
	}
	else if (type == ROTATION_DRIVING)
	{
		double accum_theta = (init_v + cons_v * (ct - start_time + plus_time));
		accum_theta -= n_rev * 2.0 * M_PI;
	//	if (accum_theta >= (n_rev + 1) * 2.0 * M_PI)
		//	n_rev++;
	//	accum_theta -= n_rev * 2.0 * M_PI;
			
		if (start_time > ct || end_time < ct) {
			v = theta - (init_v + 0.0 * ct);
		}
		else
			v = theta - accum_theta;
		//last_ce = v;
		rhs(sr) = mul * v;
	}
}

void xDrivingConstraint::ConstraintDerivative(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct, double mul)
{
	rhs(sr) = mul * cons_v;
}

void xDrivingConstraint::ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct)
{
	unsigned int i = kconst->IndexBaseBody() * xModel::OneDOF();
	unsigned int j = kconst->IndexActionBody() * xModel::OneDOF();
	vector3d ri = new_vector3d(q(i + 0), q(i + 1), q(i + 2));
	vector3d rj = new_vector3d(q(j + 0), q(j + 1), q(j + 2));
	euler_parameters ei = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ej = new_euler_parameters(q(j + 3), q(j + 4), q(j + 5), q(j + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	int ic = (kconst->IndexBaseBody() - 1) * xModel::OneDOF();
	int jc = (kconst->IndexActionBody() - 1) * xModel::OneDOF();
	if (type == TRANSLATION_DRIVING)
	{
		vector3d dist = (rj + Aj * kconst->Spj()) - (ri + Ai * kconst->Spi());// kconst->CurrentDistance();VEC3D fdij = kconst->CurrentDistance();
		vector3d hi = Ai * kconst->Hi();// im->toGlobal(kconst->h_i());
		if (i)
		{
			vector4d v = dist * BMatrix(ei, kconst->Hi()) - hi * BMatrix(ei, kconst->Spi());
			lhs.insert(sr, ic, -hi, v);
		} 
		if (j)
		{
			vector4d v = hi * BMatrix(ej, kconst->Spj());
			lhs.insert(sr, jc, hi, v);
		}
	}
	else if (type == ROTATION_DRIVING)
	{
 		vector3d fi = Ai * kconst->Fi();
 		vector3d fj = Aj * kconst->Fj();
 		vector3d gi = Ai * kconst->Gi();
		bool isSin = false;
		theta = xUtilityFunctions::RelativeAngle(udrl, theta, n_rev, gi, fi, fj, isSin);
	//	std::cout << "driving angle : " << theta << std::endl;
		vector4d D1 = kconst->relative_rotation_constraintJacobian_e_i(theta, ei, ej, fj);
		vector4d D2 = kconst->relative_rotation_constraintJacobian_e_j(theta, ei, ej, fi, gi);
		vector3d zv = new_vector3d(0.0, 0.0, 0.0);
		if (i)
			lhs.insert(sr, ic, zv, D1);
		if (j)
			lhs.insert(sr, jc, zv, D2);
	}
	srow = sr;
}

void xDrivingConstraint::DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& q_1, double* lm, unsigned int sr, double mul, double ct)
{
	double theta0 = theta;
	unsigned int idx = kconst->IndexBaseBody() * xModel::OneDOF();
	unsigned int jdx = kconst->IndexActionBody() * xModel::OneDOF();
	int ic = (kconst->IndexBaseBody() - 1) * xModel::OneDOF();
	int jc = (kconst->IndexActionBody() - 1) * xModel::OneDOF();
	if (type == ROTATION_DRIVING)
	{
		double d[8] = { 0, };
		vector4d eqi[8] = { 0, };
		vector4d eqj[8] = { 0, };
		double q0[8] = { q_1(idx + 3), q_1(idx + 4), q_1(idx + 5), q_1(idx + 6), q_1(jdx + 3), q_1(jdx + 4), q_1(jdx + 5), q_1(jdx + 6) };
		double q1[8] = { q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6), q(jdx + 3), q(jdx + 4), q(jdx + 5), q(jdx + 6) };
		// i body differential
		euler_parameters ei = new_euler_parameters(q1[0], q1[1], q1[2], q1[3]);
		euler_parameters ej = new_euler_parameters(q1[4], q1[5], q1[6], q1[7]);
		matrix33d Ai = GlobalTransformationMatrix(ei);
		matrix33d Aj = GlobalTransformationMatrix(ej);
		vector3d Fi = Ai * kconst->Fi();
		vector3d Fj = Aj * kconst->Fj();
		vector3d Gi = Ai * kconst->Gi();
		vector4d eq_i = kconst->relative_rotation_constraintJacobian_e_i(theta0, ei, ej, Fj);
		vector4d eq_j = kconst->relative_rotation_constraintJacobian_e_j(theta0, ei, ej, Fi, Gi);
		for (int i = 0; i < 8; i++)
		{
			euler_parameters e[2] = { ei, ej };
			//euler_parameters ej = ej0;
			*((&e[0].e0) + i) = q0[i];
			d[i] = 1.0 / (q1[i] - q0[i]);
			Ai = GlobalTransformationMatrix(e[0]);
			Aj = GlobalTransformationMatrix(e[1]);
			Fi = Ai * kconst->Fi();
			Fj = Aj * kconst->Fj();
			Gi = Ai * kconst->Gi();
			bool isSin = false;
			double th = xUtilityFunctions::RelativeAngle(d_udrl, theta, dn_rev, Gi, Fi, Fj, isSin);
			eqi[i] = kconst->relative_rotation_constraintJacobian_e_i(th, e[0], e[1], Fj);
			eqj[i] = kconst->relative_rotation_constraintJacobian_e_j(th, e[0], e[1], Fi, Gi);
		}
		matrix44d m44;
		if (idx)
		{
			m44 =
			{
				(eq_i.x - eqi[0].x) * d[0], (eq_i.x - eqi[1].x) * d[1], (eq_i.x - eqi[2].x) * d[2], (eq_i.x - eqi[3].x) * d[3],
				(eq_i.y - eqi[0].y) * d[0], (eq_i.y - eqi[1].y) * d[1], (eq_i.y - eqi[2].y) * d[2], (eq_i.y - eqi[3].y) * d[3],
				(eq_i.z - eqi[0].z) * d[0], (eq_i.z - eqi[1].z) * d[1], (eq_i.z - eqi[2].z) * d[2], (eq_i.z - eqi[3].z) * d[3],
				(eq_i.w - eqi[0].w) * d[0], (eq_i.w - eqi[1].w) * d[1], (eq_i.w - eqi[2].w) * d[2], (eq_i.w - eqi[3].w) * d[3]
			};
			lhs.plus(ic, ic, mul * lm[0] * m44);
		}
		if (jdx)
		{
			m44 =
			{
				(eq_j.x - eqj[4].x) * d[4], (eq_j.x - eqj[5].x) * d[5], (eq_j.x - eqj[6].x) * d[6], (eq_j.x - eqj[7].x) * d[7],
				(eq_j.y - eqj[4].y) * d[4], (eq_j.y - eqj[5].y) * d[5], (eq_j.y - eqj[6].y) * d[6], (eq_j.y - eqj[7].y) * d[7],
				(eq_j.z - eqj[4].z) * d[4], (eq_j.z - eqj[5].z) * d[5], (eq_j.z - eqj[6].z) * d[6], (eq_j.z - eqj[7].z) * d[7],
				(eq_j.w - eqj[4].w) * d[4], (eq_j.w - eqj[5].w) * d[5], (eq_j.w - eqj[6].w) * d[6], (eq_j.w - eqj[7].w) * d[7]
			};
			lhs.plus(jc, jc, mul * lm[0] * m44);
		}
		if (idx && jdx)
		{
			m44 =
			{
				(eq_i.x - eqj[4].x) * d[4], (eq_i.x - eqj[5].x) * d[5], (eq_i.x - eqj[6].x) * d[6], (eq_i.x - eqj[7].x) * d[7],
				(eq_i.y - eqj[4].y) * d[4], (eq_i.y - eqj[5].y) * d[5], (eq_i.y - eqj[6].y) * d[6], (eq_i.y - eqj[7].y) * d[7],
				(eq_i.z - eqj[4].z) * d[4], (eq_i.z - eqj[5].z) * d[5], (eq_i.z - eqj[6].z) * d[6], (eq_i.z - eqj[7].z) * d[7],
				(eq_i.w - eqj[4].w) * d[4], (eq_i.w - eqj[5].w) * d[5], (eq_i.w - eqj[6].w) * d[6], (eq_i.w - eqj[7].w) * d[7]
			};
			lhs.plus(ic, jc, mul * lm[0] * m44);

			m44 =
			{
				(eq_j.x - eqj[0].x) * d[0], (eq_j.x - eqj[1].x) * d[1], (eq_j.x - eqj[2].x) * d[2], (eq_j.x - eqj[3].x) * d[3],
				(eq_j.y - eqj[0].y) * d[0], (eq_j.y - eqj[1].y) * d[1], (eq_j.y - eqj[2].y) * d[2], (eq_j.y - eqj[3].y) * d[3],
				(eq_j.z - eqj[0].z) * d[0], (eq_j.z - eqj[1].z) * d[1], (eq_j.z - eqj[2].z) * d[2], (eq_j.z - eqj[3].z) * d[3],
				(eq_j.w - eqj[0].w) * d[0], (eq_j.w - eqj[1].w) * d[1], (eq_j.w - eqj[2].w) * d[2], (eq_j.w - eqj[3].w) * d[3]
			};
			lhs.plus(jc, ic, mul * lm[0] * m44);
		}
	}
	else if (type == TRANSLATION_DRIVING)
	{
		vector3d ri = new_vector3d(q(idx + 0), q(idx + 1), q(idx + 2));
		vector3d rj = new_vector3d(q(jdx + 0), q(jdx + 1), q(jdx + 2));
		euler_parameters ei = new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
		euler_parameters ej = new_euler_parameters(q(jdx + 3), q(jdx + 4), q(jdx + 5), q(jdx + 6));
		matrix33d Ai = GlobalTransformationMatrix(ei);
		matrix33d Aj = GlobalTransformationMatrix(ej);
		vector3d dij = (rj + Aj * kconst->Spj()) - (ri + Ai * kconst->Spi());
		kconst->dot_2_differentialJacobian(lhs, kconst->Hi(), kconst->Spi(), kconst->Spj(), dij, ei, ej, mul * lm[0]);
	}
}

xKinematicConstraint::kinematicConstraint_result xDrivingConstraint::GetStepResult(
	unsigned int part, double ct, xVectorD& q, xVectorD& qd, double* L, unsigned int sr)
{
	unsigned int i = kconst->IndexBaseBody() * xModel::OneDOF();
	unsigned int j = kconst->IndexActionBody() * xModel::OneDOF();
	vector3d ri = new_vector3d(q(i + 0), q(i + 1), q(i + 2));
	vector3d rj = new_vector3d(q(j + 0), q(j + 1), q(j + 2));
	euler_parameters ei = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ej = new_euler_parameters(q(j + 3), q(j + 4), q(j + 5), q(j + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	int ic = (kconst->IndexBaseBody() - 1) * xModel::OneDOF();
	int jc = (kconst->IndexActionBody() - 1) * xModel::OneDOF();
	xKinematicConstraint::kinematicConstraint_result kcr = { 0, };
	double* lm = L + sr;
	if (type == TRANSLATION_DRIVING)
	{
		vector3d dist = (rj + Aj * kconst->Spj()) - (ri + Ai * kconst->Spi());// kconst->CurrentDistance();VEC3D fdij = kconst->CurrentDistance();
		vector3d hi = Ai * kconst->Hi();// im->toGlobal(kconst->h_i());
		if (i)
		{
			vector4d v = dist * BMatrix(ei, kconst->Hi()) - hi * BMatrix(ei, kconst->Spi());
			kcr.iaforce = lm[0] * -hi;
			kcr.irforce = 0.5 * LMatrix(ei) * (lm[0] * v);
			//lhs.insert(sr, ic, -hi, v);
		}
		if (j)
		{
			vector4d v = hi * BMatrix(ej, kconst->Spj());
			kcr.jaforce = lm[0] * hi;
			kcr.jrforce = 0.5 * LMatrix(ej) * (lm[0] * v);
			//lhs.insert(sr, jc, hi, v);
		}
	}
	else if (type == ROTATION_DRIVING)
	{
		vector3d fi = Ai * kconst->Fi();
		vector3d fj = Aj * kconst->Fj();
		vector3d gi = Ai * kconst->Gi();
		//theta = xUtilityFunctions::RelativeAngle(ct, udrl, theta, n_rev, gi, fi, fj);
		//	std::cout << "driving angle : " << theta << std::endl;
		vector4d D1 = kconst->relative_rotation_constraintJacobian_e_i(theta, ei, ej, fj);
		vector4d D2 = kconst->relative_rotation_constraintJacobian_e_j(theta, ei, ej, fi, gi);
		vector3d zv = new_vector3d(0.0, 0.0, 0.0);
		if (i)
		{
			kcr.iaforce = zv;
			kcr.irforce = 0.5 * LMatrix(ei) * (lm[0] * D1);
		}
			//lhs.insert(sr, ic, zv, D1);
		if (j)
		{
			kcr.jaforce = zv;
			kcr.jrforce = 0.5 * LMatrix(ej) * (lm[0] * D2);
		}
			//lhs.insert(sr, jc, zv, D2);*/
	}
	kcr.location = ri + Ai * kconst->Spi();
	//kcrs.push_back(kcr);
	//nr_part++;
	return kcr;
}

void xDrivingConstraint::DerivateEquation(xVectorD& v, xVectorD& q, xVectorD& qd, int sr, double ct, double mul)
{
	if (ct < start_time)
		return;
	unsigned int i = kconst->IndexBaseBody() * xModel::OneDOF();
	unsigned int j = kconst->IndexActionBody() * xModel::OneDOF();
	vector3d ri = new_vector3d(q(i + 0), q(i + 1), q(i + 2));
	vector3d rj = new_vector3d(q(j + 0), q(j + 1), q(j + 2));
	euler_parameters ei = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ej = new_euler_parameters(q(j + 3), q(j + 4), q(j + 5), q(j + 6));
	matrix33d Ai = Transpose(GlobalTransformationMatrix(ei));
	matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix33d TAi = Transpose(kconst->BaseBody()->TransformationMatrix());
	//matrix33d Aj = kconst->ActionBody()->TransformationMatrix();
	vector3d fi = kconst->Fi();
	vector3d fj = kconst->Fj();
	vector3d gi = kconst->Gi();
	if (sr < 0)	sr = srow;
	if (type == ROTATION_DRIVING)
	{
		double s;
		double c;
		vector3d af = TAi * (Aj*fj);
		s = dot(gi, af);
		c = dot(fi, af);
		if (abs(c) >= abs(s))
		{
			v(sr) = -mul * cons_v;
		}
		else if (abs(s) > abs(c))
		{
			v(sr) = mul * cons_v;
		}
	}
	else if (type == TRANSLATION_DRIVING)
	{
		v(sr) = mul * cons_v;
	}
}

//double xDrivingConstraint::RelativeAngle(double ct, vector3d& gi, vector3d& fi, vector3d& fj)
//{
//	double df = dot(fi, fj);
//	double dg = dot(gi, fj);
//	double a = acos(dot(fi, fj));
//	double b = asin(dg);
//	double a_deg = a * 180 / M_PI;
//	double b_deg = b * 180 / M_PI;
//	double stheta = 0.0;
//	int p_udrl = udrl;
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
//	if(stheta >= 2.0 * M_PI)
//		stheta -= 2.0 * M_PI;
//	//std::cout << "stheta : " << stheta << std::endl;
//	return stheta;// xUtilityFunctions::AngleCorrection(prad, stheta);
//}

