// xdynamics_object.cpp : DLL 응용 프로그램을 위해 내보낸 함수를 정의합니다.
//

//#include "stdafx.h"
#include "xdynamics_object/xPointMass.h"
#include "xdynamics_manager/xModel.h"


xPointMass::xPointMass(xShapeType _s)
	: xObject(_s)
	, nr_part(0)
	//, pmrs(NULL)
	, initial_data(NULL)
{
	//memset(&id, 0, sizeof(*this));
	memset(&nr_part, 0, 564);
	ep.e0 = 1.0;
	setupTransformationMatrix();
	stop_condition = { 0, };
}

xPointMass::xPointMass(std::string _name, xShapeType _s)
	: xObject(_name, _s)
	, nr_part(0)
	//, pmrs(NULL)
	, initial_data(NULL)
	, mass(0)
{
	//name = _name;
	//unsigned int sc = sizeof(name);
	//unsigned int s = sizeof(*this);
	memset(&nr_part, 0, 564);
	//wsprintfW(name, TEXT("%s"), _name);
	ep.e0 = 1.0;
	setupTransformationMatrix();
	stop_condition = { 0, };
}

xPointMass::xPointMass(const xPointMass& xpm)
	: xObject(*this)
	//, pmrs(NULL)
	, initial_data(NULL)
	, mass(xpm.Mass())
	, syme_inertia(xpm.SymetricInertia())
	, diag_inertia(xpm.DiaginalInertia())
	, inertia(xpm.Inertia())
	, pos(xpm.Position())
	, vel(xpm.Velocity())
	, acc(xpm.Acceleration())
	, omega(xpm.AngularVelocity())
	, alpha(xpm.AngularAcceleration())
	, af(xpm.AxialForce())
	, am(xpm.AxialMoment())
	, cf(xpm.ContactForce())
	, cm(xpm.ContactMoment())
	, hf(xpm.HydroForce())
	, hm(xpm.HydroMoment())
	, ep(xpm.EulerParameters())
	, ev(xpm.DEulerParameters())
	, ea(xpm.DDEulerParameters())
	, A(xpm.TransformationMatrix())
{
	id = xpm.xpmIndex();
	//wsprintfW(name, TEXT("%s"), xpm.Name());
}

xPointMass::~xPointMass()
{
	//if (pmrs) delete[] pmrs; pmrs = NULL;
	if (initial_data) delete[] initial_data; initial_data = NULL;
}

void xPointMass::setXpmIndex(int idx)
{
	index = idx;
}

void xPointMass::setMass(double _mass)
{
	mass = _mass;
}
void xPointMass::setPosition(double x, double y, double z)
{
	pos = new_vector3d(x, y, z);
}
void xPointMass::setVelocity(double x, double y, double z)
{
	vel = new_vector3d(x, y, z);
}
void xPointMass::setAcceleration(double x, double y, double z)
{
	acc = new_vector3d(x, y, z);
}
void xPointMass::setEulerParameters(double e0, double e1, double e2, double e3)
{
	ep = new_euler_parameters(e0, e1, e2, e3);
}
void xPointMass::setDEulerParameters(double e0, double e1, double e2, double e3)
{
	ev = new_euler_parameters(e0, e1, e2, e3);
}
void xPointMass::setDDEulerParameters(double e0, double e1, double e2, double e3)
{
	ea = new_euler_parameters(e0, e1, e2, e3);
}
void xPointMass::setSymetricInertia(double xy, double xz, double yz)
{
	syme_inertia = new_vector3d(xy, xz, yz);
}
void xPointMass::setDiagonalInertia(double xx, double yy, double zz)
{
	diag_inertia = new_vector3d(xx, yy, zz);
}
void xPointMass::setAngularVelocity(double x, double y, double z)
{
	omega = new_vector3d(x, y, z);
}
void xPointMass::setAngularAcceleration(double x, double y, double z)
{
	alpha = new_vector3d(x, y, z);
}
void xPointMass::setAxialForce(double x, double y, double z)
{
	af = new_vector3d(x, y, z);
}
void xPointMass::setAxialMoment(double x, double y, double z)
{
	am = new_vector3d(x, y, z);
}
void xPointMass::setContactForce(double x, double y, double z)
{
	cf = new_vector3d(x, y, z);
}
void xPointMass::setContactMoment(double x, double y, double z)
{
	cm = new_vector3d(x, y, z);
}
void xPointMass::setHydroForce(double x, double y, double z)
{
	hf = new_vector3d(x, y, z);
}
void xPointMass::setHydroMoment(double x, double y, double z)
{
	hm = new_vector3d(x, y, z);
}

void xPointMass::setEulerParameterMoment(double m0, double m1, double m2, double m3)
{
	em = new_vector4d(m0, m1, m2, m3);
}

void xPointMass::setStopCondition(xSimulationStopType sst, xComparisonType ct, double value)
{
	stop_condition = { true, sst, ct, value };
}

void xPointMass::addContactForce(double x, double y, double z)
{
	cf += new_vector3d(x, y, z);
}

void xPointMass::addContactMoment(double x, double y, double z)
{
	cm += new_vector3d(x, y, z);
}

void xPointMass::addHydroForce(double x, double y, double z)
{
	hf += new_vector3d(x, y, z);
}

void xPointMass::addHydroMoment(double x, double y, double z)
{
	hm += new_vector3d(x, y, z);
}

void xPointMass::addAxialForce(double x, double y, double z)
{
	af += new_vector3d(x, y, z);
}

void xPointMass::addAxialMoment(double x, double y, double z)
{
	am += new_vector3d(x, y, z);
}

void xPointMass::addEulerParameterMoment(double m0, double m1, double m2, double m3)
{
	em += new_vector4d(m0, m1, m2, m3);
}

int xPointMass::xpmIndex() const
{
	return index;
}

// Declaration get functions
double xPointMass::Mass() const
{
	return mass;
}

matrix33d xPointMass::Inertia() const
{
	return inertia;
}

vector3d xPointMass::Position() const
{
	return pos;
}
vector3d xPointMass::Velocity() const
{
	return vel;
}
vector3d xPointMass::Acceleration() const
{
	return acc;
}
euler_parameters xPointMass::EulerParameters() const
{
	return ep;
}
euler_parameters xPointMass::DEulerParameters() const
{
	return ev;
}
euler_parameters xPointMass::DDEulerParameters() const
{
	return ea;
}
vector3d xPointMass::SymetricInertia() const
{
	return syme_inertia;
}
vector3d xPointMass::DiaginalInertia() const
{
	return diag_inertia;
}
vector3d xPointMass::AngularVelocity() const
{
	return omega;
}
vector3d xPointMass::AngularAcceleration() const
{
	return alpha;
}
vector3d xPointMass::AxialForce() const
{
	return af;
}
vector3d xPointMass::AxialMoment() const
{
	return am;
}
vector3d xPointMass::ContactForce() const
{
	return cf;
}
vector3d xPointMass::ContactMoment() const
{
	return cm;
}
vector3d xPointMass::HydroForce() const
{
	return hf;
}
vector3d xPointMass::HydroMoment() const
{
	return hm;
}

vector4d xPointMass::EulerParameterMoment() const
{
	return em;
}

QVector<xPointMass::pointmass_result>* xPointMass::XPointMassResultPointer()
{
	return &pmrs;
}

// Declaration operate functions
void xPointMass::setupTransformationMatrix()
{
	A.a00 = 2 * (ep.e0*ep.e0 + ep.e1*ep.e1 - 0.5);	A.a01 = 2 * (ep.e1*ep.e2 - ep.e0*ep.e3);		A.a02 = 2 * (ep.e1*ep.e3 + ep.e0*ep.e2);
	A.a10 = 2 * (ep.e1*ep.e2 + ep.e0*ep.e3);		A.a11 = 2 * (ep.e0*ep.e0 + ep.e2*ep.e2 - 0.5);	A.a12 = 2 * (ep.e2*ep.e3 - ep.e0*ep.e1);
	A.a20 = 2 * (ep.e1*ep.e3 - ep.e0*ep.e2);		A.a21 = 2 * (ep.e2*ep.e3 + ep.e0*ep.e1);		A.a22 = 2 * (ep.e0*ep.e0 + ep.e3*ep.e3 - 0.5);
}

void xPointMass::setupInertiaMatrix()
{
	inertia.a00 = diag_inertia.x;
	inertia.a11 = diag_inertia.y;
	inertia.a22 = diag_inertia.z;
	inertia.a01 = inertia.a10 = syme_inertia.x;
	inertia.a02 = inertia.a20 = syme_inertia.y;
	inertia.a12 = inertia.a21 = syme_inertia.z;
}

matrix33d xPointMass::TransformationMatrix() const 
{
	return A;
}

vector3d xPointMass::toLocal(const vector3d& v)
{
	vector3d tv;
	tv.x = A.a00*v.x + A.a10*v.y + A.a20*v.z;
	tv.y = A.a01*v.x + A.a11*v.y + A.a21*v.z;
	tv.z = A.a02*v.x + A.a12*v.y + A.a22*v.z;
	return tv;
}
vector3d xPointMass::toGlobal(const vector3d& v)
{
	vector3d tv;
	tv.x = A.a00*v.x + A.a01*v.y + A.a02*v.z;
	tv.y = A.a10*v.x + A.a11*v.y + A.a12*v.z;
	tv.z = A.a20*v.x + A.a21*v.y + A.a22*v.z;
	return tv;
}

void xPointMass::AllocResultMomory(unsigned int _s)
{
	if (pmrs.size())
	{
		pmrs.clear();// delete[] pmrs;
		//pmrs = NULL;
	}
	//pmrs = new pointmass_result[_s];
}

void xPointMass::setZeroAllForce()
{
	memset(&af.x, 0, sizeof(double) * 22);
}

void xPointMass::SaveStepResult(unsigned int part, double time, xVectorD& q, xVectorD& qd, xVectorD& qdd)
{
	int i = xpmIndex() * xModel::OneDOF();
	int _i = (xpmIndex() - 1) * xModel::OneDOF();
	euler_parameters e = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
	euler_parameters ed = new_euler_parameters(qd(i + 3), qd(i + 4), qd(i + 5), qd(i + 6));
	euler_parameters edd = new_euler_parameters(qdd(_i + 3), qdd(_i + 4), qdd(_i + 5), qdd(_i + 6));
	vector3d aa = 2.0 * GMatrix(e) * edd;// m->getEP().G() * m->getEA();
	vector3d av = 2.0 * GMatrix(e) * ed;// m->getEP().G() * m->getEV();
	pointmass_result pmr = 
	{
		time,
		new_vector3d(q(i + 0), q(i + 1), q(i + 2)),
		new_vector3d(qd(i + 0), qd(i + 1), qd(i + 2)),
		new_vector3d(qdd(_i + 0), qdd(_i + 1), qdd(_i + 2)),
		av, aa, af, am, cf, cm, hf, hm, e, ed, edd
	};
	pmrs.push_back(pmr);
	nr_part++;
}

void xPointMass::SaveStepResult(double time)
{
	vector3d aa = 2.0 * GMatrix(ep) * ea;// m->getEP().G() * m->getEA();
	vector3d av = 2.0 * GMatrix(ep) * ev;// m->getEP().G() * m->getEV();
	pointmass_result pmr =
	{
		time,
		pos,
		vel,
		acc,
		av, aa, af, am, cf, cm, hf, hm, ep, ev, ea
	};
	pmrs.push_back(pmr);
	nr_part++;
}

void xPointMass::ExportResults(std::fstream& of)
{
	std::ofstream ofs;
	QString _path = xModel::path + xModel::name + "/" + name + ".bpm";
	//QString _path = QString(xModel::path) + QString(xModel::name) + "/" + QString(name.text()) + ".bpm";
	ofs.open(_path.toStdString(), ios::binary | ios::out);
	char t = 'p';
	int identifier = RESULT_FILE_IDENTIFIER;
	ofs.write((char*)&identifier, sizeof(int));
	ofs.write(&t, sizeof(char));
	ofs.write((char*)&nr_part, sizeof(unsigned int));
	ofs.write((char*)pmrs.data(), sizeof(pointmass_result) * nr_part);

	ofs.close();
	xLog::log("Exported : " + _path.toStdString());
	of << _path.toStdString() << endl;
	//std::cout << "Exported : " << _path.text() << std::endl;
}

void xPointMass::SetDataFromStructure(int id, xPointMassData& d)
{
	setXpmIndex(id);
	setMass(d.mass);
	setDiagonalInertia(d.ixx, d.iyy, d.izz);
	setSymetricInertia(d.ixy, d.ixz, d.iyz);
	setPosition(d.px, d.py, d.pz);
	setEulerParameters(d.e0, d.e1, d.e2, d.e3);
	setVelocity(d.vx, d.vy, d.vz);
}

void xPointMass::ImportInitialData()
{
	if (!initial_data)
		initial_data = new double[INITIAL_BUFFER_SIZE];
	memcpy(initial_data, &pos, sizeof(double) * INITIAL_BUFFER_SIZE);
}

void xPointMass::ExportInitialData()
{
	memcpy(&pos, initial_data, sizeof(double) * INITIAL_BUFFER_SIZE);
	memset(&af, 0, sizeof(double) * INITIAL_ZERO_BUFFER_SIZE);
	delete[] initial_data; initial_data = NULL;
}

void xPointMass::setNewData(xVectorD& q, xVectorD& qd)
{
	unsigned int idx = index * xModel::OneDOF();
	pos.x = q(idx + 0); pos.y = q(idx + 1); pos.z = q(idx + 2);
	ep.e0 = q(idx + 3); ep.e1 = q(idx + 4); ep.e2 = q(idx + 5); ep.e3 = q(idx + 6);
	vel.x = qd(idx + 0); vel.y = qd(idx + 1); vel.z = qd(idx + 2);
	ev.e0 = qd(idx + 3); ev.e1 = qd(idx + 4); ev.e2 = qd(idx + 5); ev.e3 = qd(idx + 6);
	af = new_vector3d(0.0, 0.0, 0.0);
	am = new_vector3d(0.0, 0.0, 0.0);
	setupTransformationMatrix();
}

void xPointMass::setNewPositionData(xVectorD& q)
{
	unsigned int idx = index * xModel::OneDOF();
	pos.x = q(idx + 0); pos.y = q(idx + 1); pos.z = q(idx + 2);
	ep.e0 = q(idx + 3); ep.e1 = q(idx + 4); ep.e2 = q(idx + 5); ep.e3 = q(idx + 6);
	//setZeroAllForce();
	setupTransformationMatrix();
}

void xPointMass::setNewVelocityData(xVectorD& qd)
{
	unsigned int idx = index * xModel::OneDOF();
	vel.x = qd(idx + 0); vel.y = qd(idx + 1); vel.z = qd(idx + 2);
	ev.e0 = qd(idx + 3); ev.e1 = qd(idx + 4); ev.e2 = qd(idx + 5); ev.e3 = qd(idx + 6);
}

bool xPointMass::checkStopCondition()
{
	double v = 0.0;
	bool is_stop = false;
	if (stop_condition.enable)
	{
		switch (stop_condition.type)
		{
		case FORCE_MAGNITUDE: v = length(af + cf + hf); break;
		}
		switch (stop_condition.comparison)
		{
		case GRATER_THAN: is_stop = v > stop_condition.value; break;
		}
	}
	return is_stop;
}

void xPointMass::UpdateByCompulsion(double dt)
{
	if (is_compulsion_moving_object)
	{
		pos += dt * const_vel;
	}
}

//bool xPointMass::checkForceStopCondition(xSimulationStopCondition xComparisonType ct, double v)
//{
//
//	return false;
//}

// void xPointMass::translation(vector3d new_pos)
// {
// 	pos = new_pos;
// }

// void xPointMass::ExportResult2ASCII(std::ifstream& ifs)
// {
// 
// }
