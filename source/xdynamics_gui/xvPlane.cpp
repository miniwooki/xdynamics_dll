#include "xvPlane.h"

xvPlane::xvPlane()
	: xvObject()
{

}

xvPlane::xvPlane(QString _name)
	: xvObject(V_PLANE, _name)
{

}

bool xvPlane::makePlaneGeometry(xPlaneObjectData& d)
{
	data = d;
	width = d.dx;
	height = d.dy;
	
	pos.x = 0.5f * (d.p1x + d.pox);
	pos.y = 0.5f * (d.p1y + d.poy);
	pos.z = 0.5f * (d.p1z + d.poz);

	p0 = new_vector3f(d.pox, d.poy, d.poz);
	p2 = new_vector3f(d.p1x, d.p1y, d.p1z);

	vector3f dr = new_vector3f(d.drx, d.dry, d.drz);
	p1 = cross(dr, p0);
	p1 = p1 / length(p1);
	double hl = sqrt(0.25 * d.dx * d.dx + 0.25 * d.dy * d.dy);
	p1 = hl * p1;

	p3 = cross(dr, p2);
	p3 = p3 / length(p3);
	p3 = hl * p3;
	display = define();
	return display;
}

void xvPlane::draw(GLenum eMode)
{
	//qDebug() << nm << " is displayed - " << glList << " - " << display;
	if (display){
		//qDebug() << nm << " is displayed - " << glList;
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glPushMatrix();
		glDisable(GL_LIGHTING);
		glColor3f(clr.redF(), clr.greenF(), clr.blueF());
		if (eMode == GL_SELECT)
			glLoadName((GLuint)ID());
	//	unsigned int idx = vcontroller::getFrame();
		//QList<geometry_motion_result>* gmr = model::rs->geometryMotionResult(nm);
// 		if (idx != 0 && gmr)
// 		{
// 
// 			// 			if (gmr->size())
// 			// 			{
// 			geometry_motion_result gm = gmr->at(idx);
// 			animationFrame(gm.p, gm.ep);
// 			//			}
// 			/*geometry_motion_result gmr = model::rs->geometryMotionResults()[nm].at(idx);
// 			animationFrame(gmr.p, gmr.ep);*/
// 		}
// 		else
// 		{
			glTranslated(pos.x, pos.y, pos.z);
			glRotated(ang.x, 0, 0, 1);
			glRotated(ang.y, 1, 0, 0);
			glRotated(ang.z, 0, 0, 1);
		//}
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
		//	qDebug() << nm << " is displayed - " << glList;
	}
}

bool xvPlane::define()
{
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	//glColor3f(0.0f, 0.0f, 1.0f);
	//glLoadName((GLuint)ID());
	glBegin(GL_QUADS);
	{
		glVertex3f(p0.x, p0.y, p0.z);
		glVertex3f(p1.x, p1.y, p1.z);
		glVertex3f(p2.x, p2.y, p2.z);
		glVertex3f(p3.x, p3.y, p3.z);
		//glVertex3f(p0.x, p0.y, p0.z);
	}
	glEnd();
	glEndList();
	//qDebug() << "glList - " << glList;
	glHiList = glGenLists(1);
	glNewList(glHiList, GL_COMPILE);
	glBegin(GL_QUADS);
	{
		glVertex3f(p0.x, p0.y, p0.z);
		glVertex3f(p1.x, p1.y, p1.z);
		glVertex3f(p2.x, p2.y, p2.z);
		glVertex3f(p3.x, p3.y, p3.z);
		//glVertex3f(p0.x, p0.y, p0.z);
	}
	glEnd();
	glEndList();
	return true;
}