/********************************************************************************
** Form generated from reading UI file 'xpass_distribution.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_XPASS_DISTRIBUTION_H
#define UI_XPASS_DISTRIBUTION_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_XPASS_DISTRIBUTION
{
public:
    QGroupBox *GB_AREA;
    QGridLayout *gridLayout_2;
    QGridLayout *gridLayout;
    QLabel *LP0;
    QLineEdit *LEP0;
    QLabel *LP1;
    QLineEdit *LEP1;
    QLabel *LP2;
    QLineEdit *LEP2;
    QLabel *LP3;
    QLineEdit *LEP3;
    QProgressBar *PROB;
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QCheckBox *CB_TXT;
    QPushButton *PB_SelectPart;
    QPushButton *PB_ANALYSIS;
    QPushButton *PB_Exit;

    void setupUi(QDialog *XPASS_DISTRIBUTION)
    {
        if (XPASS_DISTRIBUTION->objectName().isEmpty())
            XPASS_DISTRIBUTION->setObjectName(QString::fromUtf8("XPASS_DISTRIBUTION"));
        XPASS_DISTRIBUTION->resize(390, 180);
        XPASS_DISTRIBUTION->setMinimumSize(QSize(390, 180));
        XPASS_DISTRIBUTION->setMaximumSize(QSize(390, 180));
        GB_AREA = new QGroupBox(XPASS_DISTRIBUTION);
        GB_AREA->setObjectName(QString::fromUtf8("GB_AREA"));
        GB_AREA->setGeometry(QRect(10, 10, 281, 132));
        gridLayout_2 = new QGridLayout(GB_AREA);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        LP0 = new QLabel(GB_AREA);
        LP0->setObjectName(QString::fromUtf8("LP0"));

        gridLayout->addWidget(LP0, 0, 0, 1, 1);

        LEP0 = new QLineEdit(GB_AREA);
        LEP0->setObjectName(QString::fromUtf8("LEP0"));

        gridLayout->addWidget(LEP0, 0, 1, 1, 1);

        LP1 = new QLabel(GB_AREA);
        LP1->setObjectName(QString::fromUtf8("LP1"));

        gridLayout->addWidget(LP1, 1, 0, 1, 1);

        LEP1 = new QLineEdit(GB_AREA);
        LEP1->setObjectName(QString::fromUtf8("LEP1"));

        gridLayout->addWidget(LEP1, 1, 1, 1, 1);

        LP2 = new QLabel(GB_AREA);
        LP2->setObjectName(QString::fromUtf8("LP2"));

        gridLayout->addWidget(LP2, 2, 0, 1, 1);

        LEP2 = new QLineEdit(GB_AREA);
        LEP2->setObjectName(QString::fromUtf8("LEP2"));

        gridLayout->addWidget(LEP2, 2, 1, 1, 1);

        LP3 = new QLabel(GB_AREA);
        LP3->setObjectName(QString::fromUtf8("LP3"));

        gridLayout->addWidget(LP3, 3, 0, 1, 1);

        LEP3 = new QLineEdit(GB_AREA);
        LEP3->setObjectName(QString::fromUtf8("LEP3"));

        gridLayout->addWidget(LEP3, 3, 1, 1, 1);


        gridLayout_2->addLayout(gridLayout, 0, 0, 1, 1);

        PROB = new QProgressBar(XPASS_DISTRIBUTION);
        PROB->setObjectName(QString::fromUtf8("PROB"));
        PROB->setGeometry(QRect(10, 150, 381, 23));
        PROB->setValue(24);
        widget = new QWidget(XPASS_DISTRIBUTION);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setGeometry(QRect(300, 20, 85, 121));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        CB_TXT = new QCheckBox(widget);
        CB_TXT->setObjectName(QString::fromUtf8("CB_TXT"));

        verticalLayout->addWidget(CB_TXT);

        PB_SelectPart = new QPushButton(widget);
        PB_SelectPart->setObjectName(QString::fromUtf8("PB_SelectPart"));
        PB_SelectPart->setMinimumSize(QSize(0, 30));
        PB_SelectPart->setMaximumSize(QSize(16777215, 30));

        verticalLayout->addWidget(PB_SelectPart);

        PB_ANALYSIS = new QPushButton(widget);
        PB_ANALYSIS->setObjectName(QString::fromUtf8("PB_ANALYSIS"));
        PB_ANALYSIS->setMinimumSize(QSize(83, 30));
        PB_ANALYSIS->setMaximumSize(QSize(83, 30));

        verticalLayout->addWidget(PB_ANALYSIS);

        PB_Exit = new QPushButton(widget);
        PB_Exit->setObjectName(QString::fromUtf8("PB_Exit"));
        PB_Exit->setMinimumSize(QSize(83, 30));
        PB_Exit->setMaximumSize(QSize(83, 30));

        verticalLayout->addWidget(PB_Exit);


        retranslateUi(XPASS_DISTRIBUTION);

        QMetaObject::connectSlotsByName(XPASS_DISTRIBUTION);
    } // setupUi

    void retranslateUi(QDialog *XPASS_DISTRIBUTION)
    {
        XPASS_DISTRIBUTION->setWindowTitle(QApplication::translate("XPASS_DISTRIBUTION", "Dialog", nullptr));
        GB_AREA->setTitle(QApplication::translate("XPASS_DISTRIBUTION", "Rectangle area", nullptr));
        LP0->setText(QApplication::translate("XPASS_DISTRIBUTION", "point0", nullptr));
        LP1->setText(QApplication::translate("XPASS_DISTRIBUTION", "point1", nullptr));
        LP2->setText(QApplication::translate("XPASS_DISTRIBUTION", "point2", nullptr));
        LP3->setText(QApplication::translate("XPASS_DISTRIBUTION", "point3", nullptr));
        CB_TXT->setText(QApplication::translate("XPASS_DISTRIBUTION", "Export text", nullptr));
        PB_SelectPart->setText(QApplication::translate("XPASS_DISTRIBUTION", "Select part", nullptr));
        PB_ANALYSIS->setText(QApplication::translate("XPASS_DISTRIBUTION", "Analysis", nullptr));
        PB_Exit->setText(QApplication::translate("XPASS_DISTRIBUTION", "Exit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class XPASS_DISTRIBUTION: public Ui_XPASS_DISTRIBUTION {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_XPASS_DISTRIBUTION_H
