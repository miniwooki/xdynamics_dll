#ifndef XCOMMANDWINDOW_H
#define XCOMMANDWINDOW_H

#include <QMap>
#include <QDockWidget>
#include <QPlainTextEdit>

class xCommandWindow : public QDockWidget
{
	Q_OBJECT

public:
	enum mode{ CMD_INFO = 0, CMD_DEBUG, CMD_ERROR, CMD_QUESTION };
	xCommandWindow();
	xCommandWindow(QWidget* parent);
	~xCommandWindow();

	void write(mode tw, QString c);
	void printLine();
	//void addChild(tRoot, QString& _nm);

private:
	QPlainTextEdit *cmd;
};

#endif