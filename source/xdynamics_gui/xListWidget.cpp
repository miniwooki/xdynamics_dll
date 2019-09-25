#include "xListWidget.h"
#include "xdynamics_gui.h"
#include <QFrame>
#include <QHBoxLayout>
#include <QVBoxLayout>
//#include <QApplication>

xListWidget::xListWidget(QWidget* parent)
	: QDialog(parent)
	, wlist(NULL)
{

}

xListWidget::~xListWidget()
{

}

void xListWidget::setup_widget(QStringList& qsl)
{
	//QFrame *vb = new QFrame(this);
	QVBoxLayout *vlayout = new QVBoxLayout(this);
	vlayout->setMargin(0);
	//vb->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
	wlist = new QListWidget;
	wlist->addItems(qsl);
	vlayout->addWidget(wlist);
	QHBoxLayout *hlayout = new QHBoxLayout;
	QPushButton *b_ok = new QPushButton(this);
	QPushButton *b_cancel = new QPushButton(this);
	b_ok->setText("Ok");
	b_cancel->setText("Cancel");
	hlayout->addWidget(b_ok);
	hlayout->addWidget(b_cancel);
	vlayout->addLayout(hlayout);
	connect(b_ok, SIGNAL(clicked()), this, SLOT(click_ok()));
	connect(b_cancel, SIGNAL(clicked()), this, SLOT(click_cancel()));
	const QSize g_size = xdynamics_gui::XGUI()->FullWindowSize();
	//const QSize availableSize = QApplication::desktop()->availableGeometry(wlist).size();
	this->resize(g_size / 2.0);
	wlist->setSelectionMode(QAbstractItemView::ExtendedSelection);
}

QString xListWidget::get_selected_item()
{
	return selected_item;
}

QStringList xListWidget::get_selected_items()
{
	return selected_items;
}

void xListWidget::click_ok()
{
	QList<QListWidgetItem*> items = wlist->selectedItems();
	if (items.size() == 1)
	{
		QListWidgetItem *citem = wlist->currentItem();
		selected_item = citem->text();
	}
	foreach(QListWidgetItem* it, items)
	{
		selected_items.push_back(it->text());
	}

	this->close();
	this->setResult(QDialog::Accepted);
}

void xListWidget::click_cancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
}