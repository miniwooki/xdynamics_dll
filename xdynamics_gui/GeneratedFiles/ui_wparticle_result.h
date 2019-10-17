/********************************************************************************
** Form generated from reading UI file 'wparticle_result.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WPARTICLE_RESULT_H
#define UI_WPARTICLE_RESULT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QTableWidget>

QT_BEGIN_NAMESPACE

class Ui_ParticleResultDialog
{
public:
    QGridLayout *gridLayout;
    QTableWidget *TW;
    QHBoxLayout *horizontalLayout_3;
    QRadioButton *RB_Part;
    QLineEdit *LE_Part;
    QLabel *L_Part;
    QRadioButton *RB_Time;
    QLineEdit *LE_Time;
    QLabel *L_Time;
    QHBoxLayout *horizontalLayout_2;
    QCheckBox *CBPX;
    QCheckBox *CBPY;
    QCheckBox *CBPZ;
    QCheckBox *CBVX;
    QCheckBox *CBVY;
    QCheckBox *CBVZ;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_LoadTable;
    QPushButton *PB_Export;
    QPushButton *PB_Exit;

    void setupUi(QDialog *ParticleResultDialog)
    {
        if (ParticleResultDialog->objectName().isEmpty())
            ParticleResultDialog->setObjectName(QString::fromUtf8("ParticleResultDialog"));
        ParticleResultDialog->resize(627, 538);
        gridLayout = new QGridLayout(ParticleResultDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        TW = new QTableWidget(ParticleResultDialog);
        TW->setObjectName(QString::fromUtf8("TW"));

        gridLayout->addWidget(TW, 0, 0, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        RB_Part = new QRadioButton(ParticleResultDialog);
        RB_Part->setObjectName(QString::fromUtf8("RB_Part"));
        RB_Part->setChecked(true);

        horizontalLayout_3->addWidget(RB_Part, 0, Qt::AlignRight);

        LE_Part = new QLineEdit(ParticleResultDialog);
        LE_Part->setObjectName(QString::fromUtf8("LE_Part"));
        LE_Part->setMinimumSize(QSize(50, 0));
        LE_Part->setMaximumSize(QSize(50, 16777215));

        horizontalLayout_3->addWidget(LE_Part);

        L_Part = new QLabel(ParticleResultDialog);
        L_Part->setObjectName(QString::fromUtf8("L_Part"));

        horizontalLayout_3->addWidget(L_Part);

        RB_Time = new QRadioButton(ParticleResultDialog);
        RB_Time->setObjectName(QString::fromUtf8("RB_Time"));

        horizontalLayout_3->addWidget(RB_Time, 0, Qt::AlignRight);

        LE_Time = new QLineEdit(ParticleResultDialog);
        LE_Time->setObjectName(QString::fromUtf8("LE_Time"));
        LE_Time->setMaximumSize(QSize(50, 16777215));

        horizontalLayout_3->addWidget(LE_Time);

        L_Time = new QLabel(ParticleResultDialog);
        L_Time->setObjectName(QString::fromUtf8("L_Time"));

        horizontalLayout_3->addWidget(L_Time);


        gridLayout->addLayout(horizontalLayout_3, 1, 0, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        CBPX = new QCheckBox(ParticleResultDialog);
        CBPX->setObjectName(QString::fromUtf8("CBPX"));
        CBPX->setChecked(true);

        horizontalLayout_2->addWidget(CBPX, 0, Qt::AlignHCenter);

        CBPY = new QCheckBox(ParticleResultDialog);
        CBPY->setObjectName(QString::fromUtf8("CBPY"));
        CBPY->setChecked(true);

        horizontalLayout_2->addWidget(CBPY, 0, Qt::AlignHCenter);

        CBPZ = new QCheckBox(ParticleResultDialog);
        CBPZ->setObjectName(QString::fromUtf8("CBPZ"));
        CBPZ->setChecked(true);

        horizontalLayout_2->addWidget(CBPZ, 0, Qt::AlignHCenter);

        CBVX = new QCheckBox(ParticleResultDialog);
        CBVX->setObjectName(QString::fromUtf8("CBVX"));
        CBVX->setChecked(true);

        horizontalLayout_2->addWidget(CBVX, 0, Qt::AlignHCenter);

        CBVY = new QCheckBox(ParticleResultDialog);
        CBVY->setObjectName(QString::fromUtf8("CBVY"));
        CBVY->setChecked(true);

        horizontalLayout_2->addWidget(CBVY, 0, Qt::AlignHCenter);

        CBVZ = new QCheckBox(ParticleResultDialog);
        CBVZ->setObjectName(QString::fromUtf8("CBVZ"));
        CBVZ->setChecked(true);

        horizontalLayout_2->addWidget(CBVZ, 0, Qt::AlignHCenter);


        gridLayout->addLayout(horizontalLayout_2, 2, 0, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        PB_LoadTable = new QPushButton(ParticleResultDialog);
        PB_LoadTable->setObjectName(QString::fromUtf8("PB_LoadTable"));

        horizontalLayout->addWidget(PB_LoadTable);

        PB_Export = new QPushButton(ParticleResultDialog);
        PB_Export->setObjectName(QString::fromUtf8("PB_Export"));

        horizontalLayout->addWidget(PB_Export);

        PB_Exit = new QPushButton(ParticleResultDialog);
        PB_Exit->setObjectName(QString::fromUtf8("PB_Exit"));

        horizontalLayout->addWidget(PB_Exit);


        gridLayout->addLayout(horizontalLayout, 3, 0, 1, 1);


        retranslateUi(ParticleResultDialog);

        QMetaObject::connectSlotsByName(ParticleResultDialog);
    } // setupUi

    void retranslateUi(QDialog *ParticleResultDialog)
    {
        ParticleResultDialog->setWindowTitle(QApplication::translate("ParticleResultDialog", "Dialog", nullptr));
        RB_Part->setText(QApplication::translate("ParticleResultDialog", "From part", nullptr));
        L_Part->setText(QApplication::translate("ParticleResultDialog", "TextLabel", nullptr));
        RB_Time->setText(QApplication::translate("ParticleResultDialog", "From time", nullptr));
        L_Time->setText(QApplication::translate("ParticleResultDialog", "TextLabel", nullptr));
        CBPX->setText(QApplication::translate("ParticleResultDialog", "PX", nullptr));
        CBPY->setText(QApplication::translate("ParticleResultDialog", "PY", nullptr));
        CBPZ->setText(QApplication::translate("ParticleResultDialog", "PZ", nullptr));
        CBVX->setText(QApplication::translate("ParticleResultDialog", "VX", nullptr));
        CBVY->setText(QApplication::translate("ParticleResultDialog", "VY", nullptr));
        CBVZ->setText(QApplication::translate("ParticleResultDialog", "VZ", nullptr));
        PB_LoadTable->setText(QApplication::translate("ParticleResultDialog", "Load table", nullptr));
        PB_Export->setText(QApplication::translate("ParticleResultDialog", "Export", nullptr));
        PB_Exit->setText(QApplication::translate("ParticleResultDialog", "Exit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ParticleResultDialog: public Ui_ParticleResultDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WPARTICLE_RESULT_H
