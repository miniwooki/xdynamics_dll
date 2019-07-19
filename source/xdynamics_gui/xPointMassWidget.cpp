#include "xPointMassWidget.h"
#include "xdynamics_object/xPointMass.h"

wpointmass::wpointmass(QWidget* parent /* = NULL */)
	: QWidget(parent)
{
	setupUi(this);
	connect(PBConnectGeometry, SIGNAL(clicked()), this, SLOT(EnableConnectGeometry()));
	
}

wpointmass::~wpointmass()
{

}

void wpointmass::EnableConnectGeometry()
{
	isOnConnectGeomegry = true;
	emit clickEnableConnectGeometry(true);
}

void wpointmass::UpdateInformation(xPointMass* xpm)
{
	vector3d pos = xpm->Position();
	euler_parameters ep = xpm->EulerParameters();
	vector3d eangle = EulerParameterToEulerAngle(ep);
	eangle = (180 / M_PI) * eangle;
	LEGeometry->setText("None");
	LEMass->setText(QString("%1").arg(xpm->Mass()));
	LEPosition->setText(QString("%1, %2, %3").arg(pos.x).arg(pos.y).arg(pos.z));
	LEEuerParameters->setText(QString("%1, %2, %3, %4").arg(ep.e0).arg(ep.e1).arg(ep.e2).arg(ep.e3));
	LEEulerAngle->setText(QString("%1, %2, %3").arg(eangle.x).arg(eangle.y).arg(eangle.z));
	LEIxx->setText(QString("%1").arg(xpm->DiaginalInertia().x));
	LEIyy->setText(QString("%1").arg(xpm->DiaginalInertia().y));
	LEIzz->setText(QString("%1").arg(xpm->DiaginalInertia().z));
	LEIxy->setText(QString("%1").arg(xpm->SymetricInertia().x));
	LEIyz->setText(QString("%1").arg(xpm->SymetricInertia().y));
	LEIzx->setText(QString("%1").arg(xpm->SymetricInertia().z));
}