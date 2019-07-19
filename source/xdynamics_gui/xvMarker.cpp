#include "xvMarker.h"
//#include "model.h"
#include "qgl.h"

//float xvMarker::scale = 1.0;

xvMarker::xvMarker()
	: xvObject()
	, attachObject("")
	, markerScaleFlag(true)
	, isAttachMass(false)
	, scale(1.0)
{

}

xvMarker::xvMarker(QString& n, bool mcf)
	: xvObject(V_MARKER, n)
	, attachObject("")
	, markerScaleFlag(mcf)
	, isAttachMass(false)
	, scale(1.0)
{

}

xvMarker::~xvMarker()
{
	glDeleteLists(glList, 1);
}

void xvMarker::draw(GLenum eMode)
{
	if (display)
	{
		glPushMatrix();
		if (eMode == GL_SELECT){
			glLoadName((GLuint)ID());
		}
		glDisable(GL_LIGHTING);
		if (markerScaleFlag)
			glScalef(scale, scale, scale);
		bool isAnimation = (xvAnimationController::Play() || xvAnimationController::getFrame()) && isAttachMass;
		if (isAnimation)
		{
			double t = 180 / M_PI;
			unsigned int idx = xvAnimationController::getFrame();
			xPointMass::pointmass_result pmr = xvObject::pmrs->at(idx);
			glTranslated(pmr.pos.x, pmr.pos.y, pmr.pos.z);
			//xvObject::xTranslateMinMax(pmr.pos.x, pmr.pos.y, pmr.pos.z);
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
		glCallList(glList);
		glEnable(GL_LIGHTING);
		glPopMatrix();
	}
}

bool xvMarker::define(float x, float y, float z, bool isdefine_text)
{
	pos.x = x; pos.y = y; pos.z = z;
	float icon_scale = 0.08;
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_FLAT);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	{
		xVertex(0.0f, 0.0f, 0.0f);
		xVertex(icon_scale*1.0f, 0.0f, 0.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		xVertex(icon_scale*1.5f, 0.0f, 0.0f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			xVertex(icon_scale*1.0f, cos(rad)*icon_scale*0.15f, sin(rad)*icon_scale*0.15f);
		}
	}
	glEnd();

	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	{
		xVertex(0.0f, 0.0f, 0.0f);
		xVertex(0.0f, icon_scale*1.0f, 0.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		xVertex(0.0f, icon_scale*1.5f, 0.0f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			xVertex(cos(rad)*icon_scale*0.15f, icon_scale*1.0f, sin(rad)*icon_scale*0.15f);
		}
	}
	glEnd();

	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_LINES);
	{
		xVertex(0.0f, 0.0f, 0.0f);
		xVertex(0.0f, 0.0f, icon_scale*1.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		xVertex(0.0f, 0.0f, icon_scale*1.5f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			xVertex(cos(rad)*icon_scale*0.15f, sin(rad)*icon_scale*0.15f, icon_scale*1.0f);
		}
	}
	glEnd();

	glEndList();
	display = true;
	if (!isdefine_text)
	{
		xvObject::setGlobalMinMax(pos);
	//	xvObject::setGlobalMinMax(pos);
	}	
	return true;
}

bool xvMarker::define(xPointMassData& d)
{
	data = d;
// 	pos.x = static_cast<float>(data.px);
// 	pos.y = static_cast<float>(data.py);
// 	pos.z = static_cast<float>(data.pz);
	define(static_cast<float>(data.px), static_cast<float>(data.py), static_cast<float>(data.pz), false);
	isAttachMass = true;
	return true;
}
