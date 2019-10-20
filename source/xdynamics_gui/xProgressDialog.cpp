#include "xProgressDialog.h"

xProgressDialog::xProgressDialog(QDialog *parent)
	: QProgressDialog(parent)
{
	QObject::connect(&futureWatcher, &QFutureWatcher<void>::finished, this, &QProgressDialog::reset);
	QObject::connect(this, &QProgressDialog::canceled, &futureWatcher, &QFutureWatcher<void>::cancel);
	QObject::connect(&futureWatcher, &QFutureWatcher<void>::progressRangeChanged, this, &QProgressDialog::setRange);
	QObject::connect(&futureWatcher, &QFutureWatcher<void>::progressValueChanged, this, &QProgressDialog::setValue);
}

xProgressDialog::~xProgressDialog()
{

}

