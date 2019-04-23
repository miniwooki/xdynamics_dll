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
	pos = new_vector3f(0.0f, 0.0f, 0.0f);
	len = 1.f;
	r_top = 0.2f;
	r_bottom = 0.2f;
	p0[0] = 0.0f;
	p0[1] = -0.5f; 
	p0[2] = 0.0f;
	p1[0] = 0.0f;
	p1[1] = 0.5f;
	p1[2] = 0.0f;
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
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glPushMatrix();
		glDisable(GL_LIGHTING);
		glColor3f(clr.redF(), clr.greenF(), clr.blueF());
		// 		if (vcontroller::getFrame() && outPos && outRot)
		// 			//animationFrame();
		if (eMode == GL_SELECT)
			glLoadName((GLuint)ID());
		glTranslated(pos.x, pos.y, pos.z);
		glRotated(ang.x, 0, 0, 1);
		glRotated(ang.y, 1, 0, 0);
		glRotated(ang.z, 0, 0, 1);
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

	float angle = (float)(15 * (M_PI / 180));
	int iter = (int)(360 / 15);

	float h_len = len * 0.5f;
	vector3f to = new_vector3f(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
	vector3f u = to / length(to);
	//VEC3F t = to - to.dot(u) * u;
	double th = M_PI * 0.5;
	double ap = acos(u.z);
	double xi = asin(-u.y);

	if (ap > M_PI)
		ap = ap - M_PI;

// 	ang[0] = 0.f;// 180 * xi / M_PI;
// 	ang[1] = 0.f;// 180 * th / M_PI;
// 	ang[2] = 0.f;// 180 * ap / M_PI;
// 
// 	EPD ep;
// 	ep.setFromEuler(xi, th, ap);

	glPushMatrix();
	glBegin(GL_TRIANGLE_FAN);
	{
		//VEC3D p = VEC3D( 0.f, length * 0.5f, 0.f );
		//glColor3f(0.0f, 0.f, 1.f);
		//	VEC3F p2_ = ep.A() * VEC3F(p2[0], p2[1], p2[2]);
		//glVertex3f(p2[0], p2[1], p2[2]);
		//p = ep.A() * p;
		glVertex3f(p1[0], p1[1], p1[2]);
		for (int i = 0; i < iter + 1; i++){
			float rad = angle * i;
			//glColor3f(i % 2, 0.f, i % 2 + 1.f);
			vector3f q = new_vector3f(sin(rad) * r_top, cos(rad) * r_top, -0.5f * len);
			//q = ep.A() * q;
			glVertex3f(q.x, q.y, q.z);
		}
	}
	glEnd();
	glPopMatrix();
	glBegin(GL_TRIANGLE_FAN);
	{
		vector3f p = new_vector3f(0.f, -len * 0.5f, 0.f);
		//float p[3] = { 0.f, -length * 0.5f, 0.f };
		//glColor3f(0.f, 0.f, 1.f);
		//glVertex3f(p1[0], p1[1], p1[2]);
		//p = ep.A() * p;
		//glVertex3f(p.x, p.y, p.z);
		glVertex3f(p0[0], p0[1], p0[2]);
		for (int i = 0; i < iter + 1; i++){
			float rad = angle * i;
			//glColor3f(i % 2, 0.0f, i % 2 + 1.0f);
			vector3f q = new_vector3f(sin(-rad) * r_bottom, cos(-rad) * r_bottom, 0.5f * len);
			//q = ep.A() * q;
			glVertex3f(q.x, q.y, q.z);
		}
	}
	glEnd();
	glBegin(GL_QUAD_STRIP);
	{
		for (int i = 0; i < iter + 1; i++){
			float rad = angle * i;
			vector3f q1 = new_vector3f(sin(rad) * r_top, cos(rad) * r_top, len * 0.5f);
			vector3f q2 = new_vector3f(sin(rad) * r_bottom, cos(rad) * r_bottom, -len * 0.5f);
			//q1 = ep.A() * q1;
			//q2 = ep.A() * q2;
			glVertex3f(/*origin[0] + */q2.x, /*origin[1] + */q2.y, /*origin[2] + */q2.z);
			glVertex3f(/*origin[0] + */q1.x, /*origin[1] + */q1.y, /*origin[2] + */q1.z);
		}
	}
	glEnd();
	glEndList();
	return true;
}