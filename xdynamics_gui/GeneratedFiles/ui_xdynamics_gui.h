/********************************************************************************
** Form generated from reading UI file 'xdynamics_gui.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_XDYNAMICS_GUI_H
#define UI_XDYNAMICS_GUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_xdynamics_gui_mw
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QScrollArea *xIrrchlitArea;
    QWidget *scrollAreaWidgetContents;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *xdynamics_gui_mw)
    {
        if (xdynamics_gui_mw->objectName().isEmpty())
            xdynamics_gui_mw->setObjectName(QString::fromUtf8("xdynamics_gui_mw"));
        xdynamics_gui_mw->resize(600, 400);
        centralWidget = new QWidget(xdynamics_gui_mw);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        xIrrchlitArea = new QScrollArea(centralWidget);
        xIrrchlitArea->setObjectName(QString::fromUtf8("xIrrchlitArea"));
        xIrrchlitArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 598, 345));
        xIrrchlitArea->setWidget(scrollAreaWidgetContents);

        gridLayout->addWidget(xIrrchlitArea, 0, 0, 1, 1);

        xdynamics_gui_mw->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(xdynamics_gui_mw);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        xdynamics_gui_mw->setMenuBar(menuBar);
        mainToolBar = new QToolBar(xdynamics_gui_mw);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        xdynamics_gui_mw->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(xdynamics_gui_mw);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        xdynamics_gui_mw->setStatusBar(statusBar);

        retranslateUi(xdynamics_gui_mw);

        QMetaObject::connectSlotsByName(xdynamics_gui_mw);
    } // setupUi

    void retranslateUi(QMainWindow *xdynamics_gui_mw)
    {
        xdynamics_gui_mw->setWindowTitle(QApplication::translate("xdynamics_gui_mw", "xdynamics_gui", nullptr));
    } // retranslateUi

};

namespace Ui {
    class xdynamics_gui_mw: public Ui_xdynamics_gui_mw {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_XDYNAMICS_GUI_H
