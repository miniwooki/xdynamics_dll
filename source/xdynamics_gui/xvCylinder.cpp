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
		if (isplaymode && xvObject::pmrs)
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
		
		if (isSelected)
			glLineWidth(2.0);
		else
			glLineWidth(1.0);
		glCallList(glList);
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

	int one = data.thickness ? 1 : 0;
	for (unsigned int i = 0; i <= one; i++)
	{
		double t = i * data.thickness;
		int empty = data.empty;
		double bottom_t = i * (empty == 2 ? 0.0 : t);
		double top_t = i * (empty == 1 ? 0.0 : t);
		vector3d to = new_vector3d(
			data.p1x - data.p0x,
			data.p1y - data.p0y,
			data.p1z - data.p0z);
		data.length = length(to);
		double h_len = data.length * 0.5;
		vector3d u = to / length(to);
		vector3d pu = new_vector3d(-u.y, u.x, u.z);
		vector3d qu = cross(u, pu);

		matrix33d A = { u.x, pu.x, qu.x, u.y, pu.y, qu.y, u.z, pu.z, qu.z };
		double radius = data.r_top + t;
		glBegin(GL_LINE_STRIP);
		{
			for (int i = 0; i < iter + 1; i++) {
				double rad = angle * i;
				vector3d q = A * new_vector3d(h_len + top_t, sin(rad) * radius, cos(rad) * radius);
				xVertex(q.x, q.y, q.z);
			}
		}
		glEnd();
		glBegin(GL_LINE_STRIP);
		{

			for (int i = 0; i < iter + 1; i++) {
				double rad = angle * i;
				vector3d q = A * new_vector3d(-h_len - bottom_t, sin(-rad) * radius, cos(-rad) * radius);
				xVertex(q.x, q.y, q.z);
			}
		}
		glEnd();
		glBegin(GL_QUAD_STRIP);
		{
			for (int i = 0; i < iter + 1; i++) {
				double rad = angle * i;
				vector3d q1 = A * new_vector3d(h_len + top_t, sin(rad) * radius, cos(rad) * radius);
				vector3d q2 = A * new_vector3d(-h_len - bottom_t, sin(rad) * radius, cos(rad) * radius);
				xVertex(q2.x, q2.y, q2.z);
				xVertex(q1.x, q1.y, q1.z);
			}
		}
		glEnd();
	}
	
	glEndList();
	return true;
}