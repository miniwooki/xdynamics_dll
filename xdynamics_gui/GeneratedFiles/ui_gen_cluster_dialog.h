/********************************************************************************
** Form generated from reading UI file 'gen_cluster_dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
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
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_GenClusterDialog
{
public:
    QGridLayout *gridLayout_2;
    QFrame *View3D;
    QVBoxLayout *verticalLayout;
    QTableWidget *InputTable;
    QGridLayout *gridLayout;
    QLabel *L_Scale;
    QLabel *LRows;
    QSpinBox *SB_Rows;
    QLabel *LNumbers;
    QSpinBox *SB_Numbers;
    QSpinBox *SB_Scale;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Add;
    QPushButton *PB_Gen;
    QPushButton *PB_Cancel;
    QListWidget *ClusterList;
    QTextEdit *Information;

    void setupUi(QDialog *GenClusterDialog)
    {
        if (GenClusterDialog->objectName().isEmpty())
            GenClusterDialog->setObjectName(QString::fromUtf8("GenClusterDialog"));
        GenClusterDialog->resize(944, 749);
        GenClusterDialog->setMinimumSize(QSize(944, 0));
        GenClusterDialog->setMaximumSize(QSize(944, 16777215));
        gridLayout_2 = new QGridLayout(GenClusterDialog);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        View3D = new QFrame(GenClusterDialog);
        View3D->setObjectName(QString::fromUtf8("View3D"));
        View3D->setMinimumSize(QSize(500, 400));
        View3D->setMaximumSize(QSize(500, 400));

        gridLayout_2->addWidget(View3D, 0, 0, 1, 2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        InputTable = new QTableWidget(GenClusterDialog);
        InputTable->setObjectName(QString::fromUtf8("InputTable"));
        InputTable->setMinimumSize(QSize(420, 0));
        InputTable->setMaximumSize(QSize(420, 16777215));

        verticalLayout->addWidget(InputTable);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        L_Scale = new QLabel(GenClusterDialog);
        L_Scale->setObjectName(QString::fromUtf8("L_Scale"));

        gridLayout->addWidget(L_Scale, 0, 0, 1, 1);

        LRows = new QLabel(GenClusterDialog);
        LRows->setObjectName(QString::fromUtf8("LRows"));

        gridLayout->addWidget(LRows, 1, 0, 1, 1);

        SB_Rows = new QSpinBox(GenClusterDialog);
        SB_Rows->setObjectName(QString::fromUtf8("SB_Rows"));
        SB_Rows->setMaximum(1000);

        gridLayout->addWidget(SB_Rows, 1, 1, 1, 1);

        LNumbers = new QLabel(GenClusterDialog);
        LNumbers->setObjectName(QString::fromUtf8("LNumbers"));

        gridLayout->addWidget(LNumbers, 2, 0, 1, 1);

        SB_Numbers = new QSpinBox(GenClusterDialog);
        SB_Numbers->setObjectName(QString::fromUtf8("SB_Numbers"));
        SB_Numbers->setMaximum(1000000000);

        gridLayout->addWidget(SB_Numbers, 2, 1, 1, 1);

        SB_Scale = new QSpinBox(GenClusterDialog);
        SB_Scale->setObjectName(QString::fromUtf8("SB_Scale"));
        SB_Scale->setMinimum(1);
        SB_Scale->setMaximum(100);

        gridLayout->addWidget(SB_Scale, 0, 1, 1, 1);


        verticalLayout->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        PB_Add = new QPushButton(GenClusterDialog);
        PB_Add->setObjectName(QString::fromUtf8("PB_Add"));
        PB_Add->setMinimumSize(QSize(130, 0));
        PB_Add->setMaximumSize(QSize(130, 16777215));

        horizontalLayout->addWidget(PB_Add);

        PB_Gen = new QPushButton(GenClusterDialog);
        PB_Gen->setObjectName(QString::fromUtf8("PB_Gen"));
        PB_Gen->setMinimumSize(QSize(130, 0));
        PB_Gen->setMaximumSize(QSize(130, 16777215));

        horizontalLayout->addWidget(PB_Gen);

        PB_Cancel = new QPushButton(GenClusterDialog);
        PB_Cancel->setObjectName(QString::fromUtf8("PB_Cancel"));
        PB_Cancel->setMinimumSize(QSize(130, 0));
        PB_Cancel->setMaximumSize(QSize(130, 16777215));

        horizontalLayout->addWidget(PB_Cancel);


        verticalLayout->addLayout(horizontalLayout);


        gridLayout_2->addLayout(verticalLayout, 0, 2, 2, 1);

        ClusterList = new QListWidget(GenClusterDialog);
        ClusterList->setObjectName(QString::fromUtf8("ClusterList"));
        ClusterList->setMinimumSize(QSize(200, 0));
        ClusterList->setMaximumSize(QSize(200, 16777215));

        gridLayout_2->addWidget(ClusterList, 1, 0, 1, 1);

        Information = new QTextEdit(GenClusterDialog);
        Information->setObjectName(QString::fromUtf8("Information"));
        Information->setMinimumSize(QSize(294, 0));
        Information->setMaximumSize(QSize(294, 16777215));

        gridLayout_2->addWidget(Information, 1, 1, 1, 1);


        retranslateUi(GenClusterDialog);

        QMetaObject::connectSlotsByName(GenClusterDialog);
    } // setupUi

    void retranslateUi(QDialog *GenClusterDialog)
    {
        GenClusterDialog->setWindowTitle(QApplication::translate("GenClusterDialog", "Cluster generating dialog", nullptr));
        L_Scale->setText(QApplication::translate("GenClusterDialog", "Scale", nullptr));
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
