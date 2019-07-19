#include "xLineEditWidget.h"

xLineEditWidget::xLineEditWidget()
{

}

xLineEditWidget::~xLineEditWidget()
{

}

void xLineEditWidget::keyPressEvent(QKeyEvent *e)
{
	QLineEdit::keyPressEvent(e);
	emit(up_arrow_key_press());
}