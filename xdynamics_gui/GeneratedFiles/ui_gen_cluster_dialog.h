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
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
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
    QListWidget *ClusterList;
    QTextEdit *Information;
    QFrame *View3D;
    QVBoxLayout *verticalLayout_2;
    QTableWidget *InputTable;
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QLabel *L_Scale;
    QSpinBox *SB_Scale;
    QLabel *LRows;
    QSpinBox *SB_Rows;
    QLabel *label;
    QLineEdit *LE_Name;
    QLabel *LNumbers;
    QSpinBox *SB_Numbers;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_New;
    QPushButton *PB_Add;
    QPushButton *PB_Gen;
    QPushButton *PB_Cancel;

    void setupUi(QDialog *GenClusterDialog)
    {
        if (GenClusterDialog->objectName().isEmpty())
            GenClusterDialog->setObjectName(QString::fromUtf8("GenClusterDialog"));
        GenClusterDialog->resize(944, 755);
        GenClusterDialog->setMinimumSize(QSize(944, 0));
        GenClusterDialog->setMaximumSize(QSize(944, 16777215));
        gridLayout_2 = new QGridLayout(GenClusterDialog);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
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

        View3D = new QFrame(GenClusterDialog);
        View3D->setObjectName(QString::fromUtf8("View3D"));
        View3D->setMinimumSize(QSize(500, 400));
        View3D->setMaximumSize(QSize(500, 400));

        gridLayout_2->addWidget(View3D, 0, 0, 1, 2);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        InputTable = new QTableWidget(GenClusterDialog);
        InputTable->setObjectName(QString::fromUtf8("InputTable"));
        InputTable->setMinimumSize(QSize(420, 0));
        InputTable->setMaximumSize(QSize(420, 16777215));

        verticalLayout_2->addWidget(InputTable);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        L_Scale = new QLabel(GenClusterDialog);
        L_Scale->setObjectName(QString::fromUtf8("L_Scale"));

        gridLayout->addWidget(L_Scale, 0, 0, 1, 1);

        SB_Scale = new QSpinBox(GenClusterDialog);
        SB_Scale->setObjectName(QString::fromUtf8("SB_Scale"));
        SB_Scale->setMinimumSize(QSize(100, 0));
        SB_Scale->setMaximumSize(QSize(100, 16777215));
        SB_Scale->setMinimum(1);
        SB_Scale->setMaximum(100);

        gridLayout->addWidget(SB_Scale, 0, 1, 1, 1);

        LRows = new QLabel(GenClusterDialog);
        LRows->setObjectName(QString::fromUtf8("LRows"));

        gridLayout->addWidget(LRows, 0, 2, 1, 1);

        SB_Rows = new QSpinBox(GenClusterDialog);
        SB_Rows->setObjectName(QString::fromUtf8("SB_Rows"));
        SB_Rows->setMinimumSize(QSize(100, 0));
        SB_Rows->setMaximumSize(QSize(100, 16777215));
        SB_Rows->setMaximum(1000);

        gridLayout->addWidget(SB_Rows, 0, 3, 1, 1);

        label = new QLabel(GenClusterDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 1, 0, 1, 1);

        LE_Name = new QLineEdit(GenClusterDialog);
        LE_Name->setObjectName(QString::fromUtf8("LE_Name"));
        LE_Name->setMinimumSize(QSize(100, 0));
        LE_Name->setMaximumSize(QSize(100, 16777215));

        gridLayout->addWidget(LE_Name, 1, 1, 1, 1);

        LNumbers = new QLabel(GenClusterDialog);
        LNumbers->setObjectName(QString::fromUtf8("LNumbers"));

        gridLayout->addWidget(LNumbers, 1, 2, 1, 1);

        SB_Numbers = new QSpinBox(GenClusterDialog);
        SB_Numbers->setObjectName(QString::fromUtf8("SB_Numbers"));
        SB_Numbers->setMinimumSize(QSize(100, 0));
        SB_Numbers->setMaximumSize(QSize(100, 16777215));
        SB_Numbers->setMaximum(1000000000);

        gridLayout->addWidget(SB_Numbers, 1, 3, 1, 1);


        verticalLayout->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        PB_New = new QPushButton(GenClusterDialog);
        PB_New->setObjectName(QString::fromUtf8("PB_New"));
        PB_New->setMinimumSize(QSize(100, 0));
        PB_New->setAutoDefault(false);

        horizontalLayout->addWidget(PB_New);

        PB_Add = new QPushButton(GenClusterDialog);
        PB_Add->setObjectName(QString::fromUtf8("PB_Add"));
        PB_Add->setMinimumSize(QSize(100, 0));
        PB_Add->setMaximumSize(QSize(130, 16777215));
        PB_Add->setAutoDefault(false);

        horizontalLayout->addWidget(PB_Add);

        PB_Gen = new QPushButton(GenClusterDialog);
        PB_Gen->setObjectName(QString::fromUtf8("PB_Gen"));
        PB_Gen->setMinimumSize(QSize(100, 0));
        PB_Gen->setMaximumSize(QSize(130, 16777215));
        PB_Gen->setAutoDefault(false);

        horizontalLayout->addWidget(PB_Gen);

        PB_Cancel = new QPushButton(GenClusterDialog);
        PB_Cancel->setObjectName(QString::fromUtf8("PB_Cancel"));
        PB_Cancel->setMinimumSize(QSize(100, 0));
        PB_Cancel->setMaximumSize(QSize(130, 16777215));
        PB_Cancel->setAutoDefault(false);

        horizontalLayout->addWidget(PB_Cancel);


        verticalLayout->addLayout(horizontalLayout);


        verticalLayout_2->addLayout(verticalLayout);


        gridLayout_2->addLayout(verticalLayout_2, 0, 2, 2, 1);


        retranslateUi(GenClusterDialog);

        QMetaObject::connectSlotsByName(GenClusterDialog);
    } // setupUi

    void retranslateUi(QDialog *GenClusterDialog)
    {
        GenClusterDialog->setWindowTitle(QApplication::translate("GenClusterDialog", "Cluster generating dialog", nullptr));
        L_Scale->setText(QApplication::translate("GenClusterDialog", "Scale", nullptr));
        LRows->setText(QApplication::translate("GenClusterDialog", "Rows", nullptr));
        label->setText(QApplication::translate("GenClusterDialog", "Name", nullptr));
        LNumbers->setText(QApplication::translate("GenClusterDialog", "Numbers", nullptr));
        PB_New->setText(QApplication::translate("GenClusterDialog", "New cluster", nullptr));
        PB_Add->setText(QApplication::translate("GenClusterDialog", "Add", nullptr));
        PB_Gen->setText(QApplication::translate("GenClusterDialog", "Generate model", nullptr));
        PB_Cancel->setText(QApplication::translate("GenClusterDialog", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class GenClusterDialog: public Ui_GenClusterDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GEN_CLUSTER_DIALOG_H
