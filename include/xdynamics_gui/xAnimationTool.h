#ifndef XANIMATIONTOOL_H
#define XANIMATIONTOOL_H

#include <QtWidgets/QToolBar>
#include <QtWidgets/QSlider>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>

class xGLWidget;

class xAnimationTool : public QToolBar
{
	Q_OBJECT
public:
	enum { ANIMATION_GO_BEGIN = 0, ANIMATION_PREVIOUS_2X, ANIMATION_PREVIOUS_1X, ANIMATION_PLAY_BACK, ANIMATION_INIT, ANIMATION_PLAY, ANIMATION_FORWARD_1X, ANIMATION_FORWARD_2X, ANIMATION_GO_END };
	xAnimationTool(QWidget* parent);
	~xAnimationTool();

	void setup(xGLWidget* gl);
	void update(int pt);
//	QSlider* AnimationSlider();
	
private slots:
	void xAnimationOperator();
	void xChangeAnimationFrame();
	void xChangeAnimationSlider();

private:
	void OnAnimationPlay(int tp);
	void OnAnimationPause(int tp);

	xGLWidget* xgl;
	QSlider* HSlider;
	QLineEdit* LEframe;
	QLabel* Lframe;
	QLineEdit* LETimes;
	QList<QAction*> myAnimationActions;
};

#endif