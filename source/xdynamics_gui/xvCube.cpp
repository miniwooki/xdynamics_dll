#include "xvCube.h"
#include <QTextStream>

xvCube::xvCube()
	: xvObject()
{
	/*origin[0] = origin[1] = origin[2] = 0.f;*/
	setIndexList();
	setNormalList();
}

xvCube::xvCube(QString& _name)
	: xvObject(V_CUBE, _name)
{
	setIndexList();
	setNormalList();
}

void xvCube::setIndexList()
{
	indice[0] = 0; indice[1] = 2; indice[2] = 4; indice[3] = 6;
	indice[4] = 1; indice[5] = 3; indice[6] = 2; indice[7] = 0;
	indice[8] = 5; indice[9] = 7; indice[10] = 6; indice[11] = 4;
	indice[12] = 3; indice[13] = 5; indice[14] = 4; indice[15] = 2;
	indice[16] = 7; indice[17] = 1; indice[18] = 0; indice[19] = 6;
	indice[20] = 1; indice[21] = 7; indice[22] = 5; indice[23] = 3;
}

void xvCube::setNormalList()
{
	normal[0] = 0; normal[1] = -1.0; normal[2] = 0;
	normal[3] = -1.0; normal[4] = 0.0; normal[5] = 0.0;
	normal[6] = 1.0; normal[7] = 0.0; normal[8] = 0.0;
	normal[9] = 0.0; normal[10] = 0.0; normal[11] = 1.0;
	normal[12] = 0.0; normal[13] = 0.0; normal[14] = -1.0;
	normal[15] = 0.0; normal[16] = 1.0; normal[17] = 0.0;
}

bool xvCube::makeCubeGeometry(xCubeObjectData& d)
{
	pos.x = (d.p1x + d.p0x) * 0.5f;
	pos.y = (d.p1y + d.p0y) * 0.5f;
	pos.z = (d.p1z + d.p0z) * 0.5f;
	vertice[0] = d.p0x - pos.x; vertice[1] = d.p0y - pos.y;	vertice[2] = d.p0z - pos.z;
	vertice[3] = d.p0x - pos.x;	vertice[4] = d.p1y - pos.y; vertice[5] = d.p0z - pos.z;
	vertice[6] = d.p0x - pos.x; vertice[7] = d.p0y - pos.y;	vertice[8] = d.p1z - pos.z;
	vertice[9] = d.p0x - pos.x;	vertice[10] = d.p1y - pos.y; vertice[11] = d.p1z - pos.z;
	vertice[12] = d.p1x - pos.x; vertice[13] = d.p0y - pos.y; vertice[14] = d.p1z - pos.z;
	vertice[15] = d.p1x - pos.x; vertice[16] = d.p1y - pos.y; vertice[17] = d.p1z - pos.z;
	vertice[18] = d.p1x - pos.x; vertice[19] = d.p0y - pos.y; vertice[20] = d.p0z - pos.z;
	vertice[21] = d.p1x - pos.x; vertice[22] = d.p1y - pos.y; vertice[23] = d.p0z - pos.z;
	
	//cpos = pos0;
	display = this->define();
	data = d;
	return true;
}

void xvCube::draw(GLenum eMode)
{
	if (display){
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glPushMatrix();
		//glDisable(GL_LIGHTING);
		glColor4f(clr.redF(), clr.greenF(), clr.blueF(), blend_alpha);
// 		if (vcontroller::getFrame() && outPos && outRot)
// 			//animationFrame();
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
			glTranslated(pos.x, pos.y, pos.z);
			glRotated(ang.x, 0, 0, 1);
			glRotated(ang.y, 1, 0, 0);
			glRotated(ang.z, 0, 0, 1);
		}
		
		//glCallList(glList);
		if (isSelected)
		{
			glLineWidth(2.0);
			//glLineStipple(5, 0x5555);
			//glEnable(GL_LINE_STIPPLE);
			//glColor3f(1.0f, 0.0f, 0.0f);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			//glCallList(glHiList);
			//glDisable(GL_LINE_STIPPLE);
		}
		else
			glLineWidth(1.0);
		glCallList(glList);
		glPopMatrix();
		//glEnable(GL_LIGHTING);
		//glDisable(GL_BLEND);
	}
}

bool xvCube::define()
{
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_SMOOTH);

	glBegin(GL_QUADS);
	for (int i(0); i < 6; i++){
		int *id = &indice[i * 4];
		glNormal3f(normal[i * 3 + 0], normal[i * 3 + 1], normal[i * 3 + 2]);
		glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
		glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
		glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
		glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	}
	glEnd();
	glEndList();


	//glHiList = glGenLists(1);
	//glNewList(glHiList, GL_COMPILE);
	////	glColor3f(1.0, 0.0, 0.0);
	////glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//glBegin(GL_LINE_LOOP);
	//int *id = &indice[0];
	//glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
	//glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
	//glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
	//glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	//glEnd();
	//glBegin(GL_LINE_LOOP);
	//id = &indice[5 * 4];
	//glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
	//glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
	//glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
	//glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	//glEnd();
	//glBegin(GL_LINES);
	//glVertex3f(vertice[0], vertice[1], vertice[2]);
	//glVertex3f(vertice[3], vertice[4], vertice[5]);

	//glVertex3f(vertice[6], vertice[7], vertice[8]);
	//glVertex3f(vertice[9], vertice[10], vertice[11]);

	//glVertex3f(vertice[12], vertice[13], vertice[14]);
	//glVertex3f(vertice[15], vertice[16], vertice[17]);

	//glVertex3f(vertice[18], vertice[19], vertice[20]);
	//glVertex3f(vertice[21], vertice[22], vertice[23]);
	//glEnd();
	//glEndList();
	return true;
}