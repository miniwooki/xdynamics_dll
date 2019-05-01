#include "xvObject.h"

int xvObject::count = -1;

xvObject::xvObject()
	: //outPos(NULL)
//	, outRot(NULL)
	 select_cube(NULL)
//	, vot(VIEW_OBJECT)
	, drawingMode(GL_LINE)
	, type(V_OBJECT)
	, pmrs(NULL)
	, blend_alpha(0.5f)
	//, m_type(NO_MATERIAL)
	, display(false)
	, isSelected(false)
// 	, ixx(0), iyy(0), izz(0)
// 	, ixy(0), ixz(0), iyz(0)
// 	, mass(0)
// 	, vol(0)
{
	count++;
	id = count;
//	clr = colors[count];
	pos = new_vector3f(0.0f, 0.0f, 0.0f);
	ang = new_vector3f(0.0f, 0.0f, 0.0f);
}

xvObject::xvObject(Type tp, QString _name)
	: name(_name)
// 	, outPos(NULL)
// 	, outRot(NULL)
	, select_cube(NULL)
//	, vot(VIEW_OBJECT)
	, drawingMode(GL_LINE)
	, type(tp)
	, pmrs(NULL)
	, blend_alpha(1.0f)
//	, m_type(NO_MATERIAL)
	, display(false)
	, isSelected(false)
{
	count++;
	id = count;
	pos = new_vector3f(0.0f, 0.0f, 0.0f);
	ang = new_vector3f(0.0f, 0.0f, 0.0f);
}

xvObject::~xvObject()
{
	count--;
}

void xvObject::setName(QString n)
{
	name = n;
}

void xvObject::setConnectedMassName(QString n)
{
	connected_mass_name = n;
}

void xvObject::setAngle(float x, float y, float z)
{
	ang.x = x; ang.y = y; ang.z = z;
}

void xvObject::setPosition(float x, float y, float z)
{
	pos.x = x; pos.y = y; pos.z = z;
}

void xvObject::bindPointMassResultsPointer(xPointMass::pointmass_result* _pmrs)
{
	pmrs = _pmrs;
}

void xvObject::setSelected(bool b)
{
	isSelected = b;
}

// void xvObject::setColor(color_type ct)
// {
// 	clr = colors[(int)ct];
// }
// 
// void xvObject::msgBox(QString ch, QMessageBox::Icon ic)
// {
// 	QMessageBox msg;
// 	msg.setIcon(ic);
// 	msg.setText(ch);
// 	msg.exec();
// }
// 
// void xvObject::setResultData(unsigned int n)
// {
// 	if (!outPos)
// 		outPos = new VEC3D[n];
// 	if (!outRot)
// 		outRot = new EPD[n];
// }
// 
// void xvObject::insertResultData(unsigned int i, VEC3D& p, EPD& r)
// {
// 	outPos[i] = p;
// 	outRot[i] = r;
// }
// 
// void xvObject::copyCoordinate(GLuint _coord)
// {
// 	coord = _coord;
// }
// 
// void xvObject::updateView(VEC3D& _pos, VEC3D& _ang)
// {
// 	pos0 = _pos;
// 	//ang0 = _ang;
// 	if (select_cube)
// 		select_cube->updateView(_pos, ang0);
// }

// void xvObject::animationFrame(VEC3D& p, EPD& ep)
//{
//	//unsigned int f = vcontroller::getFrame();
//	glTranslated(p.x, p.y, p.z);
//	VEC3D e = ep2e(ep);
//	double xi = (e.x * 180) / M_PI;
//	double th = (e.y * 180) / M_PI;
//	double ap = (e.z * 180) / M_PI;
//	double diff = xi + ap;
//	glRotated(xi, 0, 0, 1);
//	glRotated(th, 1, 0, 0);
//	glRotated(ap, 0, 0, 1);
//}