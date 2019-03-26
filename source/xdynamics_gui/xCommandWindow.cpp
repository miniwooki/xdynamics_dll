#include "xCommandWindow.h"

xCommandWindow::xCommandWindow()
{

}

xCommandWindow::xCommandWindow(QWidget* parent)
	: QDockWidget(parent)
{
	cmd = new QPlainTextEdit;
	setWidget(cmd);
	cmd->setReadOnly(true);
}

xCommandWindow::~xCommandWindow()
{
	if (cmd) delete cmd; cmd = NULL;
}

void xCommandWindow::write(tWriting tw, QString c)
{
	QString t;
	switch (tw)
	{
	case CMD_INFO: t = ""; break;
	case CMD_DEBUG: t = "* "; break;
	case CMD_ERROR: t = "? "; break;
	case CMD_QUESTION: t = "!"; break;
	}
	c.prepend(t);
	//QString cc = cmd->toPlainText();
	cmd->appendPlainText(c);
	//c.clear();
}

void xCommandWindow::printLine()
{
	cmd->appendPlainText("\n-------------------------------------------------------------------------------\n");
}