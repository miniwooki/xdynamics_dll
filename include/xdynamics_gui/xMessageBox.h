#pragma once
#include <QMessageBox>
#include <QString>

class xMessageBox
{
public:
	xMessageBox();
	~xMessageBox();

	static int run(QString text, QString info = "", QMessageBox::StandardButtons buttons = QMessageBox::Default, QMessageBox::StandardButton button = QMessageBox::Default);
private:
	static QMessageBox *msg;
};