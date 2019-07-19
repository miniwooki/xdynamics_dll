#ifndef XLINEEDITWIDGET_H
#define XLINEEDITWIDGET_H

#include <QLineEdit>

class xLineEditWidget : public QLineEdit
{
	Q_OBJECT

public:
	xLineEditWidget();
	~xLineEditWidget();

signals:
	void up_arrow_key_press();

protected:
	virtual void keyPressEvent(QKeyEvent *e);
};

#endif