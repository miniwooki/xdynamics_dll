/********************************************************************************
** Form generated from reading UI file 'gen_cluster_dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GEN_CLUSTER_DIALOG_H
#define UI_GEN_CLUSTER_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTextEdit>

QT_BEGIN_NAMESPACE

class Ui_GenClusterDialog
{
public:
    QGridLayout *gridLayout;
    QTableWidget *InputTable;
    QListWidget *ClusterList;
    QTextEdit *Information;
    QLabel *LRows;
    QSpinBox *SB_Rows;
    QLabel *LNumbers;
    QPushButton *PB_Add;
    QPushButton *PB_Gen;
    QPushButton *PB_Cancel;
    QSpinBox *SB_Numbers;
    QFrame *View3D;

    void setupUi(QDialog *GenClusterDialog)
    {
        if (GenClusterDialog->objectName().isEmpty())
            GenClusterDialog->setObjectName(QString::fromUtf8("GenClusterDialog"));
        GenClusterDialog->resize(924, 713);
        GenClusterDialog->setMinimumSize(QSize(924, 0));
        GenClusterDialog->setMaximumSize(QSize(924, 16777215));
        gridLayout = new QGridLayout(GenClusterDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        InputTable = new QTableWidget(GenClusterDialog);
        InputTable->setObjectName(QString::fromUtf8("InputTable"));
        InputTable->setMinimumSize(QSize(400, 0));
        InputTable->setMaximumSize(QSize(400, 16777215));

        gridLayout->addWidget(InputTable, 0, 2, 2, 4);

        ClusterList = new QListWidget(GenClusterDialog);
        ClusterList->setObjectName(QString::fromUtf8("ClusterList"));
        ClusterList->setMinimumSize(QSize(200, 0));
        ClusterList->setMaximumSize(QSize(200, 16777215));

        gridLayout->addWidget(ClusterList, 1, 0, 4, 1);

        Information = new QTextEdit(GenClusterDialog);
        Information->setObjectName(QString::fromUtf8("Information"));
        Information->setMinimumSize(QSize(294, 0));
        Information->setMaximumSize(QSize(294, 16777215));

        gridLayout->addWidget(Information, 1, 1, 4, 1);

        LRows = new QLabel(GenClusterDialog);
        LRows->setObjectName(QString::fromUtf8("LRows"));

        gridLayout->addWidget(LRows, 2, 2, 1, 1);

        SB_Rows = new QSpinBox(GenClusterDialog);
        SB_Rows->setObjectName(QString::fromUtf8("SB_Rows"));

        gridLayout->addWidget(SB_Rows, 2, 3, 1, 3);

        LNumbers = new QLabel(GenClusterDialog);
        LNumbers->setObjectName(QString::fromUtf8("LNumbers"));

        gridLayout->addWidget(LNumbers, 3, 2, 1, 1);

        PB_Add = new QPushButton(GenClusterDialog);
        PB_Add->setObjectName(QString::fromUtf8("PB_Add"));
        PB_Add->setMinimumSize(QSize(110, 0));
        PB_Add->setMaximumSize(QSize(110, 16777215));

        gridLayout->addWidget(PB_Add, 4, 2, 1, 2);

        PB_Gen = new QPushButton(GenClusterDialog);
        PB_Gen->setObjectName(QString::fromUtf8("PB_Gen"));
        PB_Gen->setMinimumSize(QSize(100, 0));
        PB_Gen->setMaximumSize(QSize(100, 16777215));

        gridLayout->addWidget(PB_Gen, 4, 4, 1, 1);

        PB_Cancel = new QPushButton(GenClusterDialog);
        PB_Cancel->setObjectName(QString::fromUtf8("PB_Cancel"));

        gridLayout->addWidget(PB_Cancel, 4, 5, 1, 1);

        SB_Numbers = new QSpinBox(GenClusterDialog);
        SB_Numbers->setObjectName(QString::fromUtf8("SB_Numbers"));

        gridLayout->addWidget(SB_Numbers, 3, 3, 1, 3);

        View3D = new QFrame(GenClusterDialog);
        View3D->setObjectName(QString::fromUtf8("View3D"));
        View3D->setMinimumSize(QSize(500, 400));
        View3D->setMaximumSize(QSize(500, 400));

        gridLayout->addWidget(View3D, 0, 0, 1, 2);


        retranslateUi(GenClusterDialog);

        QMetaObject::connectSlotsByName(GenClusterDialog);
    } // setupUi

    void retranslateUi(QDialog *GenClusterDialog)
    {
        GenClusterDialog->setWindowTitle(QApplication::translate("GenClusterDialog", "Cluster generating dialog", nullptr));
        LRows->setText(QApplication::translate("GenClusterDialog", "Rows", nullptr));
        LNumbers->setText(QApplication::translate("GenClusterDialog", "Numbers", nullptr));
        PB_Add->setText(QApplication::translate("GenClusterDialog", "Add list", nullptr));
        PB_Gen->setText(QApplication::translate("GenClusterDialog", "Generate", nullptr));
        PB_Cancel->setText(QApplication::translate("GenClusterDialog", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class GenClusterDialog: public Ui_GenClusterDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GEN_CLUSTER_DIALOG_H
