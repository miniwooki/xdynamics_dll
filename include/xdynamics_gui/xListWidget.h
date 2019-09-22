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
	xListWidget(QWidget* parent = NULL);
	~xListWidget();

	void setup_widget(QStringList& qsl);
	QString get_selected_item();
	QStringList get_selected_items();

private:
	/*QPushButton b_ok;
	QPushButton b_cancel;*/
	QListWidget* wlist;
	QString selected_item;
	QStringList selected_items;

private slots:
	void click_ok();
	void click_cancel();
};