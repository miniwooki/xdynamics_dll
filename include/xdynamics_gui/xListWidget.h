#include <QListWidget>
#include <QDialog>
#include <QPushButton>
#include <QStringList>

QT_BEGIN_NAMESPACE
//class QPushButton;
QT_END_NAMESPACE

class xListWidget : public QDialog
{
	Q_OBJECT
public:
	xListWidget(QWidget* parent);
	~xListWidget();

	void setup_widget(QStringList& qsl);
	QString get_selected_item();

private:
	/*QPushButton b_ok;
	QPushButton b_cancel;*/
	QListWidget* wlist;
	QString selected_item;

private slots:
	void click_ok();
	void click_cancel();
};