#ifndef XRESULTWIDGET_H
#define XRESULTWIDGET_H

#include "ui_wresult.h"

class wresult : public QWidget, public Ui::wresult
{
	Q_OBJECT
public:
	wresult(QWidget* parent = NULL);
	~wresult();

	void setMinMaxValue(float min, float max);

signals:
	void clickedApplyButton();

private slots:
	void ApplyButton();
	void SelectRadioButton();

private:
	bool is_user_input;// limit_input_type;
};

#endif
