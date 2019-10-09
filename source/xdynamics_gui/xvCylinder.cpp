#include "xvCylinder.h"
#include <QTextStream>

xvCylinder::xvCylinder()
	: xvObject()
{

}

xvCylinder::xvCylinder(QString& _name)
	: xvObject(V_CYLINDER, _name)
{

}

bool xvCylinder::makeCylinderGeometry(xCylinderObjectData& d)
{
	pos.x = 0.5*(d.p1x + d.p0x);
	pos.y = 0.5*(d.p1y + d.p0y);
	pos.z = 0.5*(d.p1z + d.p0z);
	data = d;
	display = this->define();

	return true;
}

void xvCylinder::draw(GLenum eMode)
{
	if (display){
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glPushMatrix();
		//glDisable(GL_LIGHTING);
		glColor4f(clr.redF(), clr.greenF(), clr.blueF(), blend_alpha);
		if (eMode == GL_SELECT)
			glLoadName((GLuint)ID());
		bool isplaymode = (xvAnimationController::Play() || xvAnimationController::getFrame()) && xvObject::pmrs;
		if (isplaymode && xvObject::pmrs)
		{
			double t = 180 / M_PI;
			unsigned int idx = xvAnimationController::getFrame();
			xPointMass::pointmass_result pmr = xvObject::pmrs[idx];
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
		//glEnable(GL_LIGHTING);
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
	std::vector<vector3d> top_outer_point;
	std::vector<vector3d> top_inner_point;
	std::vector<vector3d> bottom_outer_point;
	std::vector<vector3d> bottom_inner_point;
	vector3d u;
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
		u = to / length(to);
		/*vector3d _e = sin(M_PI * 0.5) * u;
		double e0 = 1 - dot(_e, _e);
		euler_parameters e = { e0, _e.x, _e.y, _e.z };
		matrix33d A = GlobalTransformationMatrix(e);*/
		vector3d pu = new_vector3d(-u.y, u.z, u.x);
		vector3d qu = cross(u, pu);

		matrix33d A = { u.x, pu.x, qu.x, u.y, pu.y, qu.y, u.z, pu.z, qu.z };
		double radius = data.r_top + t;

		
		glBegin(GL_TRIANGLE_FAN);
		{
			vector3d c = A * new_vector3d(h_len + t, 0, 0);
			xVertex(c.x, c.y, c.z);
			if (i) glNormal3f(u.x, u.y, u.z);
			for (int j = 0; j < iter + 1; j++) {
				double rad = angle * j;
				vector3d q = A * new_vector3d(h_len + top_t, sin(rad) * radius, cos(rad) * radius);
				if(empty == 2) xVertex(q.x, q.y, q.z);
				if (i) top_inner_point.push_back(q);
				else top_outer_point.push_back(q);
			}
		}
		glEnd();
		
		glBegin(GL_TRIANGLE_FAN);
		{
			vector3d c = A * new_vector3d(-h_len - t, 0, 0);
			xVertex(c.x, c.y, c.z);
			if (i) glNormal3f(-u.x, -u.y, -u.z);
			for (int j = 0; j < iter + 1; j++) {
				double rad = angle * j;
				vector3d q = A * new_vector3d(-h_len - bottom_t, sin(-rad) * radius, cos(-rad) * radius);
				if(empty == 1) xVertex(q.x, q.y, q.z);
				if (i) bottom_inner_point.push_back(q);
				else bottom_outer_point.push_back(q);
			}
		}
		glEnd();
		
		
		glBegin(GL_QUAD_STRIP);
		{
			for (int j = 0; j < iter + 1; j++) {
				double rad = angle * j;
				vector3d q1 = A * new_vector3d(h_len + top_t, sin(rad) * radius, cos(rad) * radius);
				vector3d q2 = A * new_vector3d(-h_len - bottom_t, sin(rad) * radius, cos(rad) * radius);
				vector3d us = 0.5 * (q1 - (A * new_vector3d(h_len + top_t, 0, 0)));
				us = us / length(us);
				if (i) glNormal3f(us.x, us.y, us.z);
				xVertex(q2.x, q2.y, q2.z);
				xVertex(q1.x, q1.y, q1.z);
			}
		}
		glEnd();
	}
	if (data.thickness)
	{
		if (data.empty == 1)
		{
			for (unsigned int i = 0; i < top_inner_point.size(); i++)
			{
				vector3d p0 = top_outer_point[i];
				vector3d p1 = top_inner_point[i];
				glBegin(GL_LINES);
				glNormal3f(u.x, u.y, u.z);
				xVertex(p0.x, p0.y, p0.z);
				xVertex(p1.x, p1.y, p1.z);
				glEnd();
			}	
		}
		if (data.empty == 2)
		{
			for (unsigned int i = 0; i < bottom_inner_point.size(); i++)
			{
				vector3d p0 = bottom_outer_point[i];
				vector3d p1 = bottom_inner_point[i];
				glBegin(GL_LINES);
				glNormal3f(-u.x, -u.y, -u.z);
				xVertex(p0.x, p0.y, p0.z);
				xVertex(p1.x, p1.y, p1.z);
				glEnd();
			}
		}
	}
	glEndList();
	return true;
}