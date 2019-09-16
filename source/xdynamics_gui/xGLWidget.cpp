#include "xGLWidget.h"
#include <QDebug>
//#include <QKeyEvent>
#include <QMouseEvent>
#include <QLineEdit>
#include <QTimer>

#include <math.h>
#include "xvMath.h"
#include "xvGlew.h"
#include "xvPlane.h"
#include "xvCube.h"
#include "xvCylinder.h"
#include "xvAnimationController.h"
#include "xModelNavigator.h"
//#include "xColorControl.h"
#include <QShortcut>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define METER 1000

xGLWidget* ogl;

xGLWidget::xGLWidget(int argc, char** argv, QWidget *parent)
	: QGLWidget(parent)
	, vp(NULL)
	//, xcc(NULL)
	, ground_marker(NULL)
	, selectedObject(NULL)
	, zRotationFlag(false)
	, isPressRightMouseButton(false)
{
	ogl = this;
	gridSize = 0.1f;
	viewOption = 0;
	xRot = 0;
	yRot = 0;
	zRot = 0;
	unit = 1;
	trans_z = -1.0;
	//zoom = -6.16199875;
	trans_x = 0;
	moveScale = 0.01f;
	trans_y = 0;
	IconScale = 0.1;
	isSetParticle = false;
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
	setContextMenuPolicy(Qt::CustomContextMenu);
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(ShowContextMenu(const QPoint&)));
	timer->start(1);
	for (int i = 0; i < 256; i++)
	{
		keyID[i] = false;
		selected[i] = false;
	}
	LBOTTON = false;
	onZoom = false;
	onRotation = false;
	isAnimation = false;
	aFrame = 0;
	for (int i(0); i < 256; i++){
		polygons[i] = 0;
	}
	numPolygons = 0;
	votype = ALL_DISPLAY;
	xvGlew::xvGlew(argc, argv);
	setFocusPolicy(Qt::StrongFocus);
	memset(minView, 0, sizeof(float) * 3);
	memset(maxView, 0, sizeof(float) * 3);

	QShortcut *a = new QShortcut(QKeySequence("Ctrl+F"), this);
	connect(a, SIGNAL(activated()), this, SLOT(fitView()));
	//xcc = new xColorControl;
}

xGLWidget::~xGLWidget()
{
	makeCurrent();
	glDeleteLists(coordinate, 1);
	glObjectClear();
	//if (xcc) delete xcc; xcc = NULL;
}

xGLWidget* xGLWidget::GLObject()
{
	return ogl;
}

bool xGLWidget::Upload_PointMass_Results(QString fname)
{
	int begin = fname.lastIndexOf("/");
	int end = fname.lastIndexOf(".");
	QString fn = fname.mid(begin, end - begin);
	xvObject* vobj = Object(fn);
	vobj->uploadPointMassResults(fname);
	return true;
}

bool xGLWidget::Upload_DEM_Results(QStringList& sl)
{
	if (vp)
	{
		unsigned int i = 0;
		vp->setBufferMemories(sl.size());
		xvAnimationController::allocTimeMemory(sl.size());
		//xvAnimationController::allocTimeMemory(sl.size());
		xvAnimationController::setTotalFrame(sl.size()-1);
		foreach(QString s, sl)
		{
			if (!vp->UploadParticleFromFile(i++, s))
			{
				return false;
			}
		}
	}
	return true;
}

void xGLWidget::ReadSTLFile(QString& s)
{
	QFile ofs(s);
	ofs.open(QIODevice::ReadOnly);
	QTextStream qts(&ofs);
	QString ch;
	qts >> ch >> ch >> ch;
	unsigned int ntri = 0;
	while (!qts.atEnd())
	{
		qts >> ch;
		if (ch == "facet")
			ntri++;
	}
	double *vertexList = new double[ntri * 9];
	double *normalList = new double[ntri * 9];
	double x, y, z;
	double nx, ny, nz;
	ofs.close();
	ofs.setFileName(s);
	ofs.open(QIODevice::ReadOnly);
	//ofs.seekg(0, ios::beg);
	qts >> ch >> ch >> ch;
	vector3d p, q, r;// , c;
	vector3d com = new_vector3d(0.0, 0.0, 0.0);
	double _vol = 0.0;
	double min_radius = 10000.0;
	double max_radius = 0.0;
	
	for (unsigned int i = 0; i < ntri; i++)
	{
		qts >> ch >> ch >> nx >> ny >> nz;
		normalList[i * 9 + 0] = nx;
		normalList[i * 9 + 1] = ny;
		normalList[i * 9 + 2] = nz;
		normalList[i * 9 + 3] = nx;
		normalList[i * 9 + 4] = ny;
		normalList[i * 9 + 5] = nz;
		normalList[i * 9 + 6] = nx;
		normalList[i * 9 + 7] = ny;
		normalList[i * 9 + 8] = nz;
		qts >> ch >> ch;
		qts >> ch >> x >> y >> z;
		p.x = vertexList[i * 9 + 0] = 0.001 * x;
		p.y = vertexList[i * 9 + 1] = 0.001 * y;
		p.z = vertexList[i * 9 + 2] = 0.001 * z;
		vertexList[i * 9 + 0] = vertexList[i * 9 + 0];
		vertexList[i * 9 + 1] = vertexList[i * 9 + 1];
		vertexList[i * 9 + 2] = vertexList[i * 9 + 2];

		qts >> ch >> x >> y >> z;
		q.x = vertexList[i * 9 + 3] = 0.001 * x;
		q.y = vertexList[i * 9 + 4] = 0.001 * y;
		q.z = vertexList[i * 9 + 5] = 0.001 * z;
		vertexList[i * 9 + 3] = vertexList[i * 9 + 3];
		vertexList[i * 9 + 4] = vertexList[i * 9 + 4];
		vertexList[i * 9 + 5] = vertexList[i * 9 + 5];

		qts >> ch >> x >> y >> z;
		r.x = vertexList[i * 9 + 6] = 0.001 * x;
		r.y = vertexList[i * 9 + 7] = 0.001 * y;
		r.z = vertexList[i * 9 + 8] = 0.001 * z;
		vertexList[i * 9 + 6] = vertexList[i * 9 + 6];
		vertexList[i * 9 + 7] = vertexList[i * 9 + 7];
		vertexList[i * 9 + 8] = vertexList[i * 9 + 8];
		qts >> ch >> ch;
	}
	ofs.close();
	int begin = s.lastIndexOf('/');
	int end = s.lastIndexOf('.');	
	QString obj_name = s.mid(begin + 1, end - begin - 1);
	xvMeshObject *vm = new xvMeshObject(obj_name);
	vm->defineMeshObject(ntri, vertexList, normalList);
	vm->setPosition(0.0,0.0,0.0);
	delete[] vertexList;
	delete[] normalList;
	v_objs[obj_name] = vm;
	v_wobjs[vm->ID()] = (void*)vm;
}

void xGLWidget::ClearViewObject()
{
	qDeleteAll(v_objs);
	qDeleteAll(xp_objs);
	v_objs.clear();
	v_wobjs.clear();
	selectedObjects.clear();
	if (vp) delete vp; vp = NULL;
}

xvMeshObject* xGLWidget::createMeshObjectGeometry(QString& file)
{
	QFile qf(file);
	qf.open(QIODevice::ReadOnly);
	unsigned int ns;
	QString obj_name;
	int material = -1;
	double loc[3] = { 0, };
	unsigned int ntriangle = 0;
	qf.read((char*)&ns, sizeof(int));
	char* _name = new char[255];
	memset(_name, 0, sizeof(char) * 255);
	qf.read((char*)_name, sizeof(char) * ns);
	obj_name.sprintf("%s", _name);
	qf.read((char*)&material, sizeof(int));
	qf.read((char*)&loc[0], sizeof(double) * 3);
	qf.read((char*)&ntriangle, sizeof(unsigned int));
	double* _vertex = new double[ntriangle * 9];
	double* _normal = new double[ntriangle * 9];
	qf.read((char*)_vertex, sizeof(double) * ntriangle * 9);
	qf.read((char*)_normal, sizeof(double) * ntriangle * 9);
	xvMeshObject *vm = new xvMeshObject(obj_name);
	vm->defineMeshObject(ntriangle, _vertex, _normal);
	vm->setPosition(loc[0], loc[1], loc[2]);
	qf.close();
	delete[] _name;
	delete[] _vertex;
	delete[] _normal;
	v_objs[obj_name] = vm;
	v_wobjs[vm->ID()] = (void*)vm;
	return vm;
//	return vm;
}

void xGLWidget::createCubeGeometry(QString& _name, xCubeObjectData& d)
{
	xvCube *vc = new xvCube(_name);
	vc->makeCubeGeometry(d);
	v_objs[_name] = vc;
	v_wobjs[vc->ID()] = (void*)vc;
}

void xGLWidget::createPlaneGeometry(QString& _name, xPlaneObjectData& d)
{
	xvPlane *vpp = new xvPlane(_name);
	vpp->makePlaneGeometry(d);
	v_objs[_name] = vpp;
	v_wobjs[vpp->ID()] = (void*)vpp;
	//qDebug() << p->Name() << " is made";
}

void xGLWidget::createCylinderGeometry(QString& _name, xCylinderObjectData& d)
{
	xvCylinder *vcy = new xvCylinder(_name);
	vcy->makeCylinderGeometry(d);
	v_objs[_name] = vcy;
	v_wobjs[vcy->ID()] = (void*)vcy;

	//setMaxViewPosition(vcy->Position().x, vcy->Position().y, vcy->Position().z);
	//setMinViewPosition(vcy->Position().x, vcy->Position().y, vcy->Position().z);
}

xvParticle* xGLWidget::createParticles()
{
	if (!vp)
		vp = new xvParticle;
	return vp;
}

xvParticle * xGLWidget::createParticleObject(QString n)
{
	QMap<QString, xvParticle*>::iterator xp = xp_objs.find(n);
	if (xp == xp_objs.end())
	{
		xp_objs[n] = new xvParticle();
		return xp_objs[n];
	}
	return NULL;
}

void xGLWidget::glObjectClear()
{
	//	qDeleteAll(v_pobjs);
	qDeleteAll(v_objs);
	qDeleteAll(xp_objs);
	if (ground_marker) delete ground_marker; ground_marker = NULL;
	if (vp) delete vp; vp = NULL;
}

void xGLWidget::setXRotation(int angle)
{
	normalizeAngle(&angle);
	if (angle != xRot) {
		xRot = angle;
		emit xRotationChanged(angle);
		updateGL();
	}
}

void xGLWidget::setYRotation(int angle)
{
	normalizeAngle(&angle);
	if (angle != yRot) {
		yRot = angle;
		emit yRotationChanged(angle);
		updateGL();
	}
}

void xGLWidget::setZRotation(int angle)
{
	normalizeAngle(&angle);
	if (angle != zRot) {
		zRot = angle;
		emit zRotationChanged(angle);
		updateGL();
	}
}

void xGLWidget::ShowContextMenu(const QPoint& pos)
{
	QPoint globalPos = this->mapToGlobal(pos);
	QMenu myMenu;
	QList<QMenu*> menus;
	//selectedObject = NULL;
	//vobject* vobj = NULL;
	if (selectedObjects.size())
	{
		QString name;
		foreach(xvObject* vobj, selectedObjects)
		{
			//unsigned int id = obj->ID();
			//if (id < 1000){
			//xvObject* vobj = static_cast<xvObject*>(v_wobjs[id]);
			name = vobj->Name();
			QMenu *subMenu = new QMenu(name);
			subMenu->addAction("Select");
			subMenu->addAction("Delete");
		//	subMenu->addAction("Property");
			//subMenu->addAction("Motion");
 			if (vobj->ObjectType() == xvObject::V_POLYGON)
 			{
 				subMenu->addAction("Convert sphere");
 			}
			myMenu.addMenu(subMenu);
			menus.push_back(subMenu);
		}
		myMenu.addSeparator();
		myMenu.addAction("Wireframe");
		myMenu.addAction("Solid");
		myMenu.addAction("Shade");
	}

	QAction *selectedItem = myMenu.exec(globalPos);

	if (selectedItem){
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		QString txt = selectedItem->text();
		if (txt == "Wireframe" && selectedObject){
			selectedObject->setDrawingMode(GL_LINE);
		}
		else if (txt == "Solid" && selectedObject){
			selectedObject->setDrawingMode(GL_FILL);
		}
		else if (txt == "Shade" && selectedObject){
			glEnable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			selectedObject->setDrawingMode(GL_FILL);
		}
		else{
			QString pmenuTitle = ((QMenu*)selectedItem->parentWidget())->title();
			if (txt == "Delete"){
				actionDelete(pmenuTitle);
				xModelNavigator::NAVIGATOR()->deleteChild(xModelNavigator::SHAPE_ROOT, pmenuTitle);
			}
			else if (txt == "Select")
			{
				setSelectMarking(pmenuTitle);
				emit signalGeometrySelection(pmenuTitle);
			}
			else if (txt == "Convert sphere")
			{
				emit contextSignal(pmenuTitle, CONTEXT_CONVERT_SPHERE);
			}
// 			else if (txt == "Property"){
// 				emit contextSignal(pmenuTitle, CONTEXT_PROPERTY);
// 			}
// 			else if (txt == "Refinement")
// 			{
// 				setSelectMarking(pmenuTitle);
// 				emit contextSignal(pmenuTitle, CONTEXT_REFINEMENT);
// 			}
// 			else if (txt == "Motion")
// 			{
// 				setSelectMarking(pmenuTitle);
// 				emit contextSignal(pmenuTitle, CONTEXT_MOTION_CONDITION);
// 			}
		}
 	}
	qDeleteAll(menus);
	emit releaseOperation();
}

void xGLWidget::fitView()
{
	double ang[3] = { DEG2RAD * xRot / 16, DEG2RAD * yRot / 16, DEG2RAD * zRot / 16 };
	double maxp[3] = { 0, }, minp[3] = { 0, };
	local2global_bryant(ang[0], ang[1], ang[2], xvObject::max_vertex.x, xvObject::max_vertex.y, xvObject::max_vertex.z, maxp);
	local2global_bryant(ang[0], ang[1], ang[2], xvObject::min_vertex.x, xvObject::min_vertex.y, xvObject::min_vertex.z, minp);
	//VEC3D dp = maxp - minp;

	trans_x = -0.5 * (maxp[0] + minp[0]);// dp.x;
	trans_y = -0.5 * (maxp[1] + minp[1]);
	trans_z = -0.5 * (maxp[2] + minp[2]) - 1.1;
}

void xGLWidget::renderText(double x, double y, double z, const QString& str, QColor& c)
{
	qglColor(c);
	QGLWidget::renderText(x, y, z, str);
}

void xGLWidget::actionDelete(const QString& tg)
{
	xvObject* vobj = v_objs.take(tg);
	if (vobj)
		delete vobj;
}

void xGLWidget::initializeGL()
{
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);                              // 매끄러운 세이딩 사용
	//	glEnable(GL_CULL_FACE);                               // 후면 제거

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	//glEnable(GL_RESCALE_NORMAL); 

	GLfloat LightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat LightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat Lightemissive[] = { 0.f, 0.f, 0.f, 1.0f };
	GLfloat LightPosition[] = { 0.0f, 0.0f, 1.0f, 1.0f };
//	GLfloat LightPosition1[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
	glLightfv(GL_LIGHT0, GL_SPECULAR, LightSpecular);
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);	// Position The Light
// 	glEnable(GL_LIGHT1);
// 	glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
// 	glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
// 	glLightfv(GL_LIGHT1, GL_SPECULAR, LightSpecular);
// 	glLightfv(GL_LIGHT1, GL_POSITION, LightPosition1);	// Position The Light
	float m = 1.f / 255.f;
	GLfloat material_Ka[] = { 51.f*m, 51.f*m, 51.f*m, 1.0f };
	GLfloat material_Kd[] = { 204.f*m, 204.f*m, 204.f*m, 1.0f };
	GLfloat material_Ks[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat material_Ke[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat material_Se = 0.0f;
	
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_Ka);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_Kd);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_Ks);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, material_Ke);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_Se);

	ref_marker.setName("ref_marker");
	//ref_marker.setAttchedMass(false);
	//ref_marker.setMarkerScaleFlag(false);
	ref_marker.define(-0.85, -0.85, 0.0, true);

	ground_marker = new xvMarker(QString("ground_marker"), true);
	ground_marker->define(0.0, 0.0, 0.0);
	ground_marker->setMarkerScale(0.1);
	//ground_marker->setAttchedMass(false);
	v_wobjs[ground_marker->ID()] = (void*)ground_marker;

	protype = PERSPECTIVE_PROJECTION;
}

// void xGLWidget::makeMassCoordinate(QString& _name)
// {
// 	QMap<QString, xvObject*>::iterator it = v_objs.find(_name);
// 	xvObject* vobj = it.value();
// }

void xGLWidget::drawReferenceCoordinate()
{
	double ang[3] = { DEG2RAD * xRot / 16, DEG2RAD * yRot / 16, DEG2RAD * zRot / 16 };
	double xp[3] = { 0, }, yp[3] = { 0, }, zp[3] = { 0, };
	local2global_bryant(ang[0], ang[1], ang[2], 0.13, 0.0, 0.0, xp);
	local2global_bryant(ang[0], ang[1], ang[2], 0.0, 0.13, 0.0, yp);
	local2global_bryant(ang[0], ang[1], ang[2], 0.0, 0.0, 0.13, zp);
	renderText(xp[0] - 0.85, xp[1] - 0.85, xp[2], QString("X"), QColor(255, 0, 0));
	renderText(yp[0] - 0.85, yp[1] - 0.85, yp[2], QString("Y"), QColor(0, 255, 0));
	renderText(zp[0] - 0.85, zp[1] - 0.85, zp[2], QString("Z"), QColor(0, 0, 255));
	ref_marker.setAngle(xRot / 16, yRot / 16, zRot / 16);
	ref_marker.draw(GL_RENDER);
}

void xGLWidget::drawGroundCoordinate(GLenum eMode)
{
// 	double ang[3] = { DEG2RAD * xRot / 16, DEG2RAD * yRot / 16, DEG2RAD * zRot / 16 };
// 	renderText(0.13, 0.0, 0.0, QString("X"), QColor(255, 0, 0));
// 	renderText(0.0, 0.13, 0.0, QString("Y"), QColor(0, 255, 0));
// 	renderText(0.0, 0.0, 0.13, QString("Z"), QColor(0, 0, 255));
	ground_marker->draw(eMode);
}

// void xGLWidget::setStartingData(QMap<QString, v3epd_type> d)
// {
// 	QMapIterator<QString, v3epd_type> it(d);
// 	while (it.hasNext())
// 	{
// 		it.next();
// 		QString s = it.key();
// 		v3epd_type p = it.value();
// 		vobject* vo = Object(s);
// 		vobject* vm = Object(s + "_marker");
// 		VEC3D ang = ep2e(p.ep);
// 		double xi = (ang.x * 180) / M_PI;
// 		double th = (ang.y * 180) / M_PI;
// 		double ap = (ang.z * 180) / M_PI;
// 		if (vo)
// 		{
// 			vo->setInitialPosition(p.v3);
// 			vo->setInitialAngle(VEC3D(xi, th, ap));
// 		}
// 		if (vm)
// 		{
// 			vm->setInitialPosition(p.v3);
// 			vm->setInitialAngle(VEC3D(xi, th, ap));
// 		}
// 
// 	}
// }

xvParticle * xGLWidget::ParticleObject(QString n)
{
	QStringList l = xp_objs.keys();
	QStringList::const_iterator it = qFind(l, n);
	if (it == l.end())
		return NULL;
	return xp_objs[n];
}

xvParticle* xGLWidget::vParticles()
{
	return vp;// createParticles();
}

// xvObject* xGLWidget::getVObjectFromName(QString name)
// {
// 	return v_objs.find(name).value();
// }
// 
// vpolygon* xGLWidget::getVPolyObjectFromName(QString name)
// {
// 	return NULL;
// }

QString xGLWidget::selectedObjectName()
{
	return selectedObject ? selectedObject->Name() : "";
}

// xvObject* xGLWidget::selectedObjectWithCast()
// {
// 	if (!selectedObject)
// 		return NULL;
// 
// 	return selectedObject;
// }

GLuint xGLWidget::makePolygonObject(double* points, double* normals, int* indice, int size)
{
	GLuint list = glGenLists(1);
	glNewList(list, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	glColor3f(0.0f, 0.0f, 1.0f);
	for (int i(0); i < size; i++){
		glBegin(GL_TRIANGLES);
		{
			glNormal3dv(&normals[i * 3]);
			glVertex3dv(&points[indice[i * 3 + 0] * 3]);
			glVertex3dv(&points[indice[i * 3 + 1] * 3]);
			glVertex3dv(&points[indice[i * 3 + 2] * 3]);
		}
		glEnd();
	}
	glEndList();
	polygons[numPolygons] = list;
	numPolygons++;
	return list;
}


void xGLWidget::drawObject(GLenum eMode)
{
	glTranslatef(trans_x, trans_y, trans_z);
	glRotated(xRot / 16.0, 1.0, 0.0, 0.0);
	glRotated(yRot / 16.0, 0.0, 1.0, 0.0);
	glRotated(zRot / 16.0, 0.0, 0.0, 1.0);
	//qDebug() << xRot << " " << yRot << " " << zRot;
	drawGroundCoordinate(eMode);
	QMapIterator<QString, xvObject*> obj(v_objs);

	if (vp)
	{
		vp->draw(GL_RENDER, wHeight, protype, abs(trans_z));
	}

	foreach(xvParticle* xp, xp_objs)
	{
		xp->draw(GL_RENDER, wHeight, protype, abs(trans_z));
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	while (obj.hasNext()){
		obj.next();
		obj.value()->draw(eMode);
		//qDebug() << obj.value()->name();
	}
	glDisable(GL_BLEND);
}

xvObject* xGLWidget::setSelectMarking(QString sn)
{
	//unsigned int id = selectedIndice.at(0);
	xvObject* obj = selectedObjects[sn];// static_cast<vobject*>(v_wobjs[id]);
	obj->setSelected(true);
	selectedObject = obj;

#ifdef _DEBUG
	qDebug() << obj->Name() << " is selected.";
#endif
	return obj;
}

void xGLWidget::processHits(unsigned int uHits, unsigned int *pBuffer)
{
	unsigned int i, j;
	unsigned int uiName, *ptr;
	ptr = pBuffer;
	if (!isPressRightMouseButton)
	{
		selectedObject = NULL;
		foreach(xvObject* vobj, selectedObjects)
			vobj->setSelected(false);
	}

	// 	foreach(int v, selectedIndice)
	// 		static_cast<vobject*>(v_wobjs[v])->setSelected(false);
	if (selectedObjects.size())
		selectedObjects.clear();
	for (i = 0; i < uHits; i++){
		uiName = *ptr;
		ptr += 3;
		int idx = *ptr;
		xvObject* _vobj = Object(idx);;// <xvObject*>(v_wobjs[idx]);
		if (!_vobj) continue;
		selectedObjects[_vobj->Name()] = _vobj;
		//selectedIndice.push_back(*ptr);// selectedIndice[i] = *ptr;

		//static_cast<vobject*>(v_wobjs[id])->setSelected(true);
		ptr++;
	}
	if (selectedObjects.size() == 1)
	{
		setSelectMarking(selectedObjects.firstKey());
	}
}

void xGLWidget::setSketchSpace()
{
	QLineEdit* le = dynamic_cast<QLineEdit*>(sender());
	//sketch.space = le->text().toDouble();
}

void xGLWidget::setupParticleBufferColorDistribution(int n)
{
	//xColorControl xcc;
	//if (vp)
	//{
	//	int sframe = n < 0 ? 0 : xvAnimationController::getTotalBuffers();
	//	int cframe = xvAnimationController::getTotalBuffers();
	//	
	//	xColorControl::ColorMapType cmt = xcc->Target();
	//	if (!xcc->isUserLimitInput())
	//		xcc->setMinMax(vp->getMinValue(cmt), vp->getMaxValue(cmt));
	//	unsigned int m_np = vp->NumParticles();
	//	xcc->setLimitArray();
	//	for (int i = sframe; i <= cframe; i++)
	//	{
	//		unsigned int idx = m_np * i;
	//		float *pbuf = vp->PositionBuffers() + idx * 4;
	//		float *vbuf = vp->VelocityBuffers() + idx * 3;
	//		float *cbuf = vp->ColorBuffers() + idx * 4;
	//		for (unsigned int j = 0; j < m_np; j++)
	//		{
	//			xcc->getColorRamp(pbuf + j * 4, vbuf + j * 3, cbuf + j * 4);
	//			cbuf[j * 4 + 3] = 1.0;
	//		}
	//	}
	//}
}

void xGLWidget::sketchingMode()
{
	// 	glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
	// 	gluOrtho2D(-1.02f, 1.02f, -1.02f, 1.02f);
	// 	unsigned int numGrid = static_cast<unsigned int>(1.0f / gridSize) * 2 + 1;
	// 	glMatrixMode(GL_MODELVIEW);
	// 	glLoadIdentity();
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// 	/*glPushMatrix();*/
	//glDisable(GL_LIGHTING);
	// 	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	// 	glBegin(GL_LINES);
	// 	{
	// 		double sx = floor((sketch.sx + (sketch.ex - sketch.sx) * 0.1) * 10) * 0.1;
	// 		double ex = -sx;
	// 		double sy = floor((sketch.sy + (sketch.ey - sketch.sy) * 0.1) * 10) * 0.1;
	// 		double ey = -sy;
	// 		double lx = (floor(ex / sketch.space)) * sketch.space;
	// 		double ly = (floor(ey / sketch.space)) * sketch.space;
	// 		glPushMatrix();
	// 		glVertex3d(sx, sy, 0); glVertex3d(ex, sy, 0);
	// 		glVertex3d(ex, sy, 0); glVertex3d(ex, ey, 0);
	// 		glVertex3d(ex, ey, 0); glVertex3d(sx, ey, 0);
	// 		glVertex3d(sx, ey, 0); glVertex3d(sx, sy, 0);
	// 		glPopMatrix();
	// 		int nx = static_cast<int>((ex - sx) / sketch.space + 1e-9);
	// 		int ny = static_cast<int>((ey - sy) / sketch.space + 1e-9);
	// 		float fTmp1[16] = { 0.f, };
	// 		glGetFloatv(GL_PROJECTION_MATRIX, fTmp1);
	// 		for (int ix = 1; ix < nx; ix++)
	// 		{
	// 			double x = sx + sketch.space * ix;
	// 			glPushMatrix();
	// 			glVertex3d(x, sy, 0);
	// 			glVertex3d(x, ey, 0);
	// 			glPopMatrix();
	//  			
	// 		}
	// 		for (int iy = 1; iy < ny; iy++)
	// 		{
	// 			double y = sy + sketch.space * iy;
	// 			glPushMatrix();
	// 			glVertex3d(sx, y, 0);
	// 			glVertex3d(ex, y, 0);
	// 			glPopMatrix();
	// 		}
	// // 		for (double x = sx; x < ex; x += sketch.space){
	// // 			double rx = floor(x + 10e-9);
	// // 			for (double y = sy; y < ey; y += sketch.space){
	// // 				double ry = floor(y + 10e-9);
	// // 				glPushMatrix();
	// // 				glVertex3d(x, y, 0);
	// // 				glVertex3d(lx, y, 0);
	// // 
	// // // 				glVertex3f(x, y, 0.f);
	// // // 				glVertex3f(x, ly, 0.f);
	// // 				glPopMatrix();
	// // 			}
	// // 			glPushMatrix();
	// // 			glVertex3d(x, sy, 0);
	// // 			glVertex3d(x, ly, 0);
	// // 			glPopMatrix();
	// // 		}
	// // // 		glVertex2f(-0.98f, -0.98f);
	// // // 		glVertex2f(-0.98f, 0.98f);
	// // // 
	// // // 		glVertex2f(-0.98f, 0.98f);
	// // // 		glVertex2f(0.98f, 0.98f);
	// // // 
	// // // 		glVertex2f(0.98f, 0.98f);
	// // // 		glVertex2f(0.98f, -0.98f);
	// // // 
	// // // 		glVertex2f(0.98f, -0.98f);
	// // // 		glVertex2f(-0.98f, -0.98f);
	//  	}
	//  	glEnd();

}

float xGLWidget::GetParticleMinValueFromColorMapType()
{
	return 0.f;
	//return vp->getMinValue(xcc->Target());
}

float xGLWidget::GetParticleMaxValueFromColorMapType()
{
	return 0.f;
	//return vp->getMaxValue(xcc->Target());
}

void xGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();
	glClearColor(1.0, 1.0, 1.0, 1.0);
	gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPushMatrix();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	drawReferenceCoordinate();
	resizeGL(wWidth, wHeight);

	

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	drawObject(GL_RENDER);
	if (xvAnimationController::Play()){
		xvAnimationController::update_frame();
		emit changedAnimationFrame();
	}

	glEnable(GL_COLOR_MATERIAL);
}

void xGLWidget::resizeGL(int width, int height)
{
	wWidth = width; wHeight = height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	ratio = (GLfloat)(width) / (GLfloat)(height);
	float z = abs(trans_z);
	float c = z * tanf(30.0f * M_PI / 180.0f);
	switch (protype)
	{
	case PERSPECTIVE_PROJECTION:
		gluPerspective(60.0f, ratio, 0.01f, 1000.0f);
		break;
	case ORTHO_PROJECTION:
		if (width <= height){
			glOrtho(-1.0f * c, 1.0f * c, (-1.0f / ratio) * c, (1.0f / ratio) * c, 0.00001f, 1000.f);
		}
		else{
			glOrtho(-1.0f * c * ratio, 1.0f * c * ratio, -1.0f * c, 1.0f * c, 0.00001f, 1000.f);
		}
		break;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void xGLWidget::wheelEvent(QWheelEvent *e)
{
	QPoint  p = e->angleDelta();
	float pzoom = trans_z;
	p.y() > 0 ? trans_z -= 2.0f*moveScale : trans_z += 2.0f*moveScale;
	qDebug() << "Translation view value : " << trans_z;
	setFocusPolicy(Qt::StrongFocus);
}

void xGLWidget::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();
	if (event->button() == Qt::RightButton){
		picking(lastPos.x(), lastPos.y());
		isPressRightMouseButton = true;
	}
	if (event->button() == Qt::MiddleButton){
		onZoom = true;
	}
	if (event->button() == Qt::LeftButton){
		isPressRightMouseButton = false;
		if (keyID[82])
			onRotation = true;
		else if (!keyID[84])
			picking(lastPos.x(), lastPos.y());
	}
}

void xGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (onRotation)
		onRotation = false;

	if (onZoom)
		onZoom = false;

	if (event->button() == Qt::LeftButton){
		if (keyID[90])
			keyID[90] = false;
		if (keyID[84])
			keyID[84] = false;
		if (keyID[82])
			keyID[82] = false;
	}
}

void xGLWidget::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - lastPos.x();
	int dy = event->y() - lastPos.y();
	if (keyID[84]){
		dy > 0 ? trans_y -= 0.1f*moveScale*dy : trans_y -= 0.1f*moveScale*dy;
		dx > 0 ? trans_x += 0.1f*moveScale*dx : trans_x += 0.1f*moveScale*dx;
	}
	if (keyID[82] && onRotation) {
		if (zRotationFlag)
			setZRotation(zRot - 8 * dx);
		else
		{
			setXRotation(xRot + 8 * dy);
			setYRotation(yRot + 8 * dx);
		}
	}
	if (onZoom)
	{
		dy > 0 ? trans_z -= 0.01f*moveScale : trans_z += 0.01f*moveScale;
	}
	lastPos = event->pos();
}

void xGLWidget::picking(int x, int y)
{

	unsigned int aSelectBuffer[SELECT_BUF_SIZE];
	unsigned int uiHits;
	int aViewport[4];

	glGetIntegerv(GL_VIEWPORT, aViewport);

	glSelectBuffer(SELECT_BUF_SIZE, aSelectBuffer);
	glRenderMode(GL_SELECT);

	glInitNames();
	glPushName(0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	gluPickMatrix((double)x, (double)(aViewport[3] - y), 5.0, 5.0, aViewport);

	switch (protype)
	{
	case PERSPECTIVE_PROJECTION: gluPerspective(60.0, ratio, 0.01f, 1000.0f); break;
	case ORTHO_PROJECTION:
		if (wWidth <= wHeight)
			glOrtho(-1.f * abs(trans_z), 1.f * abs(trans_z), -1.f / ratio * abs(trans_z), 1.f / ratio * abs(trans_z), 0.01f, 1000.f);
		else
			glOrtho(-1.f * abs(trans_z) * ratio, 1.f * ratio * abs(trans_z), -1.f * abs(trans_z), 1.f * abs(trans_z), 0.01f, 1000.f);
		break;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//gluLookAt(eye[0], eye[1], eye[2], 0, 0, 0, 0, 1, 0);
	drawObject(GL_SELECT);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	uiHits = glRenderMode(GL_RENDER);
	processHits(uiHits, aSelectBuffer);
	glMatrixMode(GL_MODELVIEW);
	if (!uiHits)
		emit releaseOperation();
}

void xGLWidget::keyReleaseEvent(QKeyEvent *e)
{
	switch (e->key())
	{
	case Qt::Key_Control:
		zRotationFlag = false;
		break;
	}
}

void xGLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()){
	case Qt::Key_Up:
		verticalMovement() += 0.1f*moveScale;
		break;
	case Qt::Key_Down:
		verticalMovement() -= 0.1f*moveScale;
		break;
	case Qt::Key_Left:
		horizontalMovement() -= 0.1f*moveScale;
		break;
	case Qt::Key_Right:
		horizontalMovement() += 0.1f*moveScale;
		break;
	case Qt::Key_Plus:
		moveScale += 0.001f;
		break;
	case Qt::Key_Minus:
		moveScale -= 0.001f;
		break;
	case Qt::Key_PageUp:
// 		if (vp)
// 			vp->upParticleScale(1);
		break;
	case Qt::Key_PageDown:
// 		if (vp)
// 			vp->downParticleScale(1);
		break;
	case Qt::Key_Control:
		if (keyID[82])
			zRotationFlag = true;
		break;
	case 90:
		setKeyState(true, 90);
		break;
	case 84:
		setKeyState(true, 84);
		break;
	case 82:
		setKeyState(true, 82);
		break;
	}
	if (moveScale <= 0){
		moveScale = 0.0f;
	}
}

void xGLWidget::setMaxViewPosition(float x, float y, float z)
{
	if (x > maxView[0]) maxView[0] = x;
	if (y > maxView[1]) maxView[1] = y;
	if (z > maxView[2]) maxView[2] = z;
}

void xGLWidget::setMinViewPosition(float x, float y, float z)
{
	if (x < minView[0]) minView[0] = x;
	if (y < minView[1]) minView[1] = y;
	if (z < minView[2]) minView[2] = z;
}

void xGLWidget::normalizeAngle(int *angle)
{
	while (*angle < 0)
		*angle += 360 * 16;
	while (*angle > 360 * 16)
		*angle -= 360 * 16;
}

void xGLWidget::ChangeDisplayOption(int oid)
{
	viewOption = oid;
}

void xGLWidget::releaseSelectedObjects()
{
	foreach(xvObject* xo, selectedObjects)
	{
		xo->setSelected(false);
	}
}

xvObject* xGLWidget::Object(QString nm)
{
	QStringList l = v_objs.keys();
	QStringList::const_iterator it = qFind(l, nm);
	if (it == l.end())
		return NULL;
	return v_objs[nm];
}

xvObject* xGLWidget::Object(int id)
{
	QList<int> l = v_wobjs.keys();
	QList<int>::const_iterator it = qFind(l, id);
	if (it == l.end())
		return NULL;
	return static_cast<xvObject*>(v_wobjs[id]);
}

xvMarker* xGLWidget::makeMarker(QString n, double x, double y, double z, bool mcf)
{
	QString _name = n;
	xvMarker* vm = new xvMarker(_name, mcf);
	//vm->setAttachObject(n);
	vm->define(x, y, z);
	v_objs[_name] = vm;
	v_wobjs[vm->ID()] = (void*)vm;

	setMaxViewPosition(x, y, z);
	setMinViewPosition(x, y, z);
	return vm;
}

xvMarker* xGLWidget::makeMarker(QString& _name, xPointMassData& d)
{
	xvMarker* vm = new xvMarker(_name, true);
	//vm->setAttachObject(n);
	vm->define(d);
	v_objs[_name] = vm;
	v_wobjs[vm->ID()] = (void*)vm;

 	setMaxViewPosition(d.px, d.py, d.pz);
	setMinViewPosition(d.px, d.py, d.pz);
	return vm;
}

void xGLWidget::openResults(QStringList& fl)
{

}