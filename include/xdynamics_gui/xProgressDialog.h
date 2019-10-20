#pragma once

#include <QProgressDialog>
#include <QFutureWatcher>

class xProgressDialog : public QProgressDialog
{
	Q_OBJECT
public:
	xProgressDialog(QDialog* parent = NULL);
	~xProgressDialog();
private:
	QFutureWatcher<void> futureWatcher;
};