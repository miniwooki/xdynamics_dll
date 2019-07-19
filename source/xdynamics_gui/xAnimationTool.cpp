#include "xAnimationTool.h"
#include "xvAnimationController.h"
#include "xGLWidget.h"

xAnimationTool::xAnimationTool(QWidget* parent)
	: QToolBar(parent)
	, HSlider(NULL)
	, Lframe(NULL)
{

}

xAnimationTool::~xAnimationTool()
{

}

void xAnimationTool::setup(xGLWidget* gl)
{
	///myAnimationBar = addToolBar(tr("Animation Operations"));
	QAction *a;
	xgl = gl;
	a = new QAction(QIcon(":/Resources/icon/ani_tobegin.png"), tr("&toBegin"), this);
	a->setStatusTip(tr("Go to begin"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_GO_BEGIN, a);
	a->setData(QVariant(ANIMATION_GO_BEGIN));

	a = new QAction(QIcon(":/Resources/icon/ani_moreprevious.png"), tr("&previous2X"), this);
	a->setStatusTip(tr("previous 2X"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_PREVIOUS_2X, a);
	a->setData(QVariant(ANIMATION_PREVIOUS_2X));

	a = new QAction(QIcon(":/Resources/icon/ani_previous.png"), tr("&previous1X"), this);
	a->setStatusTip(tr("previous 1X"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_PREVIOUS_1X, a);
	a->setData(QVariant(ANIMATION_PREVIOUS_1X));

	a = new QAction(QIcon(":/Resources/icon/ani_playback.png"), tr("&animation back play"), this);
	a->setStatusTip(tr("animation back play"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_PLAY_BACK, a);
	a->setData(QVariant(ANIMATION_PLAY_BACK));

	a = new QAction(QIcon(":/Resources/icon/ani_init.png"), tr("&animation initialize"), this);
	a->setStatusTip(tr("animation initialize"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_INIT, a);
	a->setData(QVariant(ANIMATION_INIT));

	a = new QAction(QIcon(":/Resources/icon/ani_play.png"), tr("&animation play"), this);
	a->setStatusTip(tr("animation play"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_PLAY, a);
	a->setData(QVariant(ANIMATION_PLAY));

	a = new QAction(QIcon(":/Resources/icon/ani_fast.png"), tr("&forward1X"), this);
	a->setStatusTip(tr("forward 1X"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_FORWARD_1X, a);
	a->setData(QVariant(ANIMATION_FORWARD_1X));

	a = new QAction(QIcon(":/Resources/icon/ani_morefast.png"), tr("&forward2X"), this);
	a->setStatusTip(tr("forward 2X"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_FORWARD_2X, a);
	a->setData(QVariant(ANIMATION_FORWARD_2X));

	a = new QAction(QIcon(":/Resources/icon/ani_toEnd.png"), tr("&toEnd"), this);
	a->setStatusTip(tr("Go to end"));
	connect(a, SIGNAL(triggered()), this, SLOT(xAnimationOperator()));
	myAnimationActions.insert(ANIMATION_GO_END, a);
	a->setData(QVariant(ANIMATION_GO_END));

	for (int i = 0; i < myAnimationActions.size(); i++)
	{
		this->addAction(myAnimationActions.at(i));
	}
	HSlider = new QSlider(Qt::Orientation::Horizontal, this);
	HSlider->setFixedWidth(100);
	connect(HSlider, SIGNAL(valueChanged(int)), this, SLOT(xChangeAnimationFrame()));
 	connect(xgl, SIGNAL(changedAnimationFrame()), this, SLOT(xChangeAnimationFrame()));
// 	connect(gl, SIGNAL(propertySignal(QString, context_object_type)), this, SLOT(propertySlot(QString, context_object_type)));
// 	connect(db, SIGNAL(propertySignal(QString, context_object_type)), this, SLOT(propertySlot(QString, context_object_type)));
	this->addWidget(HSlider);// ->parentWidget();
	HSlider->setMaximum(xvAnimationController::getTotalBuffers());
	LEframe = new QLineEdit(this);
	LEframe->setText(QString("0"));
	LEframe->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	LEframe->setFixedWidth(50);
	LEframe->setContentsMargins(QMargins(5, 0, 0, 0));
	this->addWidget(LEframe);
	Lframe = new QLabel(this);
	Lframe->setText(QString("/ %1").arg(xvAnimationController::getTotalBuffers()));
	Lframe->setContentsMargins(QMargins(5, 0, 0, 0));
	this->addWidget(Lframe);
	LETimes = new QLineEdit(this);
	LETimes->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	LEframe->setFixedWidth(50);
	LETimes->setFixedWidth(50);
	QLabel *LTimes = new QLabel(this);
	LTimes->setText(QString("Time : "));
	LTimes->setContentsMargins(QMargins(10, 0, 0, 0));
	this->addWidget(LTimes);
	this->addWidget(LETimes);
	//myAnimationBar->hide();
}

void xAnimationTool::update(int pt)
{
	HSlider->setMaximum(pt);
	QString ch;
	QTextStream(&ch) << "/ " << pt;
	Lframe->setText(ch);
}

// QSlider* xAnimationTool::AnimationSlider()
// {
// 	return HSlider;
// }

void xAnimationTool::xAnimationOperator()
{
	QAction* a = (QAction*)sender();
	int tp = a->data().toInt();
	switch (tp)
	{
	case ANIMATION_GO_BEGIN:
		xvAnimationController::initFrame();
		xvAnimationController::off_play();
		OnAnimationPause(tp);
		xChangeAnimationFrame();
		break;
	case ANIMATION_PREVIOUS_2X:
		xvAnimationController::move2previous2x();
		OnAnimationPause(tp);
		xChangeAnimationFrame();
		break;
	case ANIMATION_PREVIOUS_1X:
		xvAnimationController::move2previous1x();
		OnAnimationPause(tp);
		xChangeAnimationFrame();
		break;
	case ANIMATION_PLAY_BACK:
		xvAnimationController::Play() ? OnAnimationPause(tp) : OnAnimationPlay(tp);
		break;
	case ANIMATION_PLAY:
		xvAnimationController::Play() ? OnAnimationPause(tp) : OnAnimationPlay(tp);
		break;
	case ANIMATION_INIT:
		OnAnimationPause(tp);
		xvAnimationController::initFrame();
		xChangeAnimationFrame();
		break;
	case ANIMATION_FORWARD_1X:
		xvAnimationController::move2forward1x();
		OnAnimationPause(tp);
		xChangeAnimationFrame();
		break;
	case ANIMATION_FORWARD_2X:
		xvAnimationController::move2forward2x();
		OnAnimationPause(tp);
		xChangeAnimationFrame();
		break;
	case ANIMATION_GO_END:
		OnAnimationPause(tp);
		xvAnimationController::moveEnd();
		xChangeAnimationFrame();
		break;
	}
}

void xAnimationTool::xChangeAnimationFrame()
{
	int cf = xvAnimationController::getFrame();
	HSlider->setValue(cf);
	QString str;
	str.sprintf("%d", cf);
	LEframe->setText(str);
	float time = xvAnimationController::getTime();
	str.clear(); str.sprintf("%f", time);
	LETimes->setText(str);
}

void xAnimationTool::xChangeAnimationSlider()
{
	if (xvAnimationController::Play())
		return;
	//OnAnimationPause(xvAnimationController);
	int value = HSlider->value();
}

void xAnimationTool::OnAnimationPlay(int tp)
{
	QAction *a = NULL;
	switch (tp)
	{
	case ANIMATION_PREVIOUS_2X:
		a = myAnimationActions[ANIMATION_PREVIOUS_2X];
		xvAnimationController::setPlayMode(ANIMATION_PREVIOUS_2X);
		break;
	case ANIMATION_FORWARD_2X:
		a = myAnimationActions[ANIMATION_FORWARD_2X];
		xvAnimationController::setPlayMode(ANIMATION_FORWARD_2X);
		break;
	case ANIMATION_PLAY:
		a = myAnimationActions[ANIMATION_PLAY];
		xvAnimationController::setPlayMode(ANIMATION_PLAY);
		break;
	case ANIMATION_PLAY_BACK:
		a = myAnimationActions[ANIMATION_PLAY_BACK];
		xvAnimationController::setPlayMode(ANIMATION_PLAY_BACK);
		break;
	}
	a->setIcon(QIcon(":/Resources/icon/ani_pause.png"));
	a->setStatusTip(tr("Restart the animation."));
	xvAnimationController::on_play();
}

void xAnimationTool::OnAnimationPause(int tp)
{
	QAction *a = NULL;
	QString icon_path;
	int previous_tp = xvAnimationController::currentPlayMode();
	switch (previous_tp)
	{
	case ANIMATION_PLAY:
		a = myAnimationActions[ANIMATION_PLAY];
		icon_path = ":/Resources/icon/ani_play.png";
		break;
	case ANIMATION_PLAY_BACK:
		a = myAnimationActions[ANIMATION_PLAY_BACK];
		icon_path = ":/Resources/icon/ani_playback.png";
		break;
	case ANIMATION_PREVIOUS_2X:
		a = myAnimationActions[ANIMATION_PREVIOUS_2X];
		icon_path = ":/Resources/icon/ani_moreprevious.png";
		break;
	case ANIMATION_FORWARD_2X:
		a = myAnimationActions[ANIMATION_FORWARD_2X];
		icon_path = ":/Resources/icon/ani_morefast.png";
		break;
	default:
		return;
	}
	if (a)
	{
		a->setIcon(QIcon(icon_path));
		a->setStatusTip(tr("Pause the animation."));
	}

	xvAnimationController::off_play();
}
