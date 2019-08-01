//#include "xdynamics_gui.h"
#include "xResultWidget.h"
#include "xColorControl.h"
//#include "xdynamics_simulation/xSimulation.h"

wresult::wresult(QWidget* parent /* = NULL */)
	: QWidget(parent)
{
	setupUi(this);
	//connect(PBSolve, SIGNAL(clicked()), this, SLOT(SolveButton()));
	connect(PB_Apply, SIGNAL(clicked()), this, SLOT(ApplyButton()));
	connect(RB_UserInput, SIGNAL(clicked()), this, SLOT(SelectRadioButton()));
	connect(RB_FromResult, SIGNAL(clicked()), this, SLOT(SelectRadioButton()));
}

wresult::~wresult()
{

}

void wresult::setMinMaxValue(float min, float max)
{
	LE_LimitMin->setText(QString("%1").arg(min));
	LE_LimitMax->setText(QString("%1").arg(max));
}

void wresult::SelectRadioButton()
{
	bool isu = RB_UserInput->isChecked();
	if (!isu)
	{
		PB_Apply->setEnabled(false);
	}
	else
	{
		PB_Apply->setEnabled(true);
	}
	isu ? xColorControl::setUserLimitInputType(true) : xColorControl::setUserLimitInputType(false);
}

void wresult::ApplyButton()
{
	//QString dt = LETimeStep->text();
	float min_v = LE_LimitMin->text().toFloat();
	float max_v = LE_LimitMax->text().toFloat();
	xColorControl::setMinMax(min_v, max_v);
	/*unsigned int st = LESaveStep->text().toUInt();
	double et = LEEndTime->text().toDouble();*/
	bool isu = RB_UserInput->isChecked();
	xColorControl::setUserLimitInputType(isu);
	emit clickedApplyButton();
}
