#include "xvCylinder.h"
#include <QTextStream>

xvCylinder::xvCylinder()
	: xvObject()
{
	/*origin[0] = origin[1] = origin[2] = 0.f;*/
// 	setIndexList();
// 	setNormalList();
}

xvCylinder::xvCylinder(QString& _name)
	: xvObject(V_CYLINDER, _name)
{
// 	setIndexList();
// 	setNormalList();
}

bool xvCylinder::makeCylinderGeometry(xCylinderObjectData& d)
{
	//pos = new_vector3f(0.0f, 0.1f, 0.25f);
// 	len = 1.f;
// 	r_top = 0.2f;
// 	r_bottom = 0.2f;
// 	p0[0] = 0.0f;
// 	p0[1] = -0.5f; 
// 	p0[2] = 0.0f;
// 	p1[0] = 0.0f;
// 	p1[1] = 0.5f;
// 	p1[2] = 0.0f;
	//xCylinderObjectData dd = { 1.0, 0.2, 0.2, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0 };
	pos.x = 0.5*(d.p1x + d.p0x);
	pos.y = 0.5*(d.p1y + d.p0y);
	pos.z = 0.5*(d.p1z + d.p0z);
	data = d;
	display = this->define();

// 	pos.x = (d.p1x + d.p0x) * 0.5f;
// 	pos.y = (d.p1y + d.p0y) * 0.5f;
// 	pos.z = (d.p1z + d.p0z) * 0.5f;
// 	vertice[0] = d.p0x - pos.x; vertice[1] = d.p0y - pos.y;	vertice[2] = d.p0z - pos.z;
// 	vertice[3] = d.p0x - pos.x;	vertice[4] = d.p1y - pos.y; vertice[5] = d.p0z - pos.z;
// 	vertice[6] = d.p0x - pos.x; vertice[7] = d.p0y - pos.y;	vertice[8] = d.p1z - pos.z;
// 	vertice[9] = d.p0x - pos.x;	vertice[10] = d.p1y - pos.y; vertice[11] = d.p1z - pos.z;
// 	vertice[12] = d.p1x - pos.x; vertice[13] = d.p0y - pos.y; vertice[14] = d.p1z - pos.z;
// 	vertice[15] = d.p1x - pos.x; vertice[16] = d.p1y - pos.y; vertice[17] = d.p1z - pos.z;
// 	vertice[18] = d.p1x - pos.x; vertice[19] = d.p0y - pos.y; vertice[20] = d.p0z - pos.z;
// 	vertice[21] = d.p1x - pos.x; vertice[22] = d.p1y - pos.y; vertice[23] = d.p0z - pos.z;
// 
// 	//cpos = pos0;
// 	display = this->define();
// 	data = d;
	return true;
}

void xvCylinder::draw(GLenum eMode)
{
	if (display){
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glPushMatrix();
		glDisable(GL_LIGHTING);
		glColor3f(clr.redF(), clr.greenF(), clr.blueF());
		if (eMode == GL_SELECT)
			glLoadName((GLuint)ID());
		bool isplaymode = (xvAnimationController::Play() || xvAnimationController::getFrame()) && xvObject::pmrs;
		if (isplaymode)
		{
			double t = 180 / M_PI;
			unsigned int idx = xvAnimationController::getFrame();
			xPointMass::pointmass_result pmr = xvObject::pmrs->at(idx);
			glTranslated(pmr.pos.x, pmr.pos.y, pmr.pos.z);
			vector3d euler = EulerParameterToEulerAngle(pmr.ep);
			glRotated(t*euler.x, 0, 0, 1);
			glRotated(t*euler.y, 1, 0, 0);
			glRotated(t*euler.z, 0, 0, 1);
		}
		else
		{
			glTranslatef(pos.x, pos.y, pos.z);
			glRotatef(ang.x, 0, 0, 1);
			glRotatef(ang.y, 1, 0, 0);
			glRotatef(ang.z, 0, 0, 1);
		}		
		glCallList(glList);
		if (isSelected)
		{
			glLineWidth(2.0);
			glLineStipple(5, 0x5555);
			glEnable(GL_LINE_STIPPLE);
			glColor3f(1.0f, 0.0f, 0.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glCallList(glHiList);
			glDisable(GL_LINE_STIPPLE);
		}
		glPopMatrix();
		glEnable(GL_LIGHTING);
	}
}

bool xvCylinder::define()
{
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_SMOOTH);

	double angle = (15 * (M_PI / 180));
	int iter = (int)(360 / 15);
	//vector3d mp = new_vector3d(0.0, 0.1, 0.25);
	
	//vector3d a = 
	vector3d to = new_vector3d(data.p1x - data.p0x, data.p1y - data.p0y, data.p1z - data.p0z);
	data.length = length(to);
	double h_len = data.length * 0.5;
	vector3d u = to / length(to);							//x
	vector3d pu = new_vector3d(-u.y, u.x, u.z);// cross(u, new_vector3d(1.0 - u.x, 1.0 - u.y, 1.0 - u.z));	//y
	vector3d qu = cross(u, pu);					
	
	//vector3d u0 = new_vector3d(0.0, 0.0, 1.0);
	//double theta = 0.5 * (M_PI * 0.5  - acos(dot(u, u0)));
	
	//double e0 = cos(theta);
	//vector3d e123 = sin(theta) * pu;
	//euler_parameters e = new_euler_parameters(e0, e123.x, e123.y, e123.z);
	//matrix33d A = GlobalTransformationMatrix(e);
	matrix33d A = { u.x, pu.x, qu.x, u.y, pu.y, qu.y, u.z, pu.z, qu.z };
	//glPushMatrix();
//	vector3d p0, p1;
// 	p0 = A * new_vector3d(h_len, 0.0, 0.0);
// 	glBegin(GL_POINT);
// 	glVertex3d(p0.x, p0.y, p0.z);
// 	glEnd();
	glBegin(GL_LINE_STRIP);
	{
 		
		for (int i = 0; i < iter + 1; i++){
			double rad = angle * i;
			vector3d q = A * new_vector3d(h_len, sin(rad) * data.r_top, cos(rad) * data.r_top);
			xVertex(q.x, q.y, q.z);
		}
	}
	glEnd();
	//glPopMatrix();
//	glPushMatrix();
// 	p1 = A * new_vector3d(-h_len, 0.0, 0.0);
// 	glBegin(GL_POINT);
// 	glVertex3d(p1.x, p1.y, p1.z);
// 	glEnd();
	glBegin(GL_LINE_STRIP);
	{
		
		for (int i = 0; i < iter + 1; i++){
			double rad = angle * i;
			vector3d q = A * new_vector3d(-h_len, sin(-rad) * data.r_bottom, cos(-rad) * data.r_bottom);
			xVertex(q.x, q.y, q.z);
		}
	}
	glEnd();
	//glPopMatrix();
// 	glPushMatrix();
	glBegin(GL_QUAD_STRIP);
	{
		for (int i = 0; i < iter + 1; i++){
			double rad = angle * i;
			vector3d q1 = A * new_vector3d(h_len, sin(rad) * data.r_top, cos(rad) * data.r_top);
			vector3d q2 = A * new_vector3d(-h_len, sin(rad) * data.r_bottom, cos(rad) * data.r_bottom);
			xVertex(q2.x, q2.y, q2.z);
			xVertex(q1.x, q1.y, q1.z);
		}
	}
	glEnd();
// 	glPopMatrix();
	glEndList();
	//xvObject::setGlobalMinMax(pos + local_min);
	//xvObject::setGlobalMinMax(pos + local_max);
	return true;
}