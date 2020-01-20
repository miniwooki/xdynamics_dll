/********************************************************************************
** Form generated from reading UI file 'wresult.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WRESULT_H
#define UI_WRESULT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wresult
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_4;
    QVBoxLayout *verticalLayout;
    QGroupBox *GBColorMap;
    QGridLayout *gridLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;
    QComboBox *CB_Target;
    QHBoxLayout *horizontalLayout;
    QRadioButton *RB_UserInput;
    QRadioButton *RB_FromResult;
    QPushButton *PB_MapData;
    QHBoxLayout *horizontalLayout_2;
    QLabel *L_LimitMin;
    QLineEdit *LE_LimitMin;
    QLabel *L_LimitMax;
    QLineEdit *LE_LimitMax;
    QPushButton *PB_Apply;

    void setupUi(QWidget *wresult)
    {
        if (wresult->objectName().isEmpty())
            wresult->setObjectName(QString::fromUtf8("wresult"));
        wresult->resize(257, 106);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wresult->sizePolicy().hasHeightForWidth());
        wresult->setSizePolicy(sizePolicy);
        wresult->setMinimumSize(QSize(0, 106));
        wresult->setMaximumSize(QSize(16777215, 106));
        gridLayout = new QGridLayout(wresult);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wresult);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_4 = new QGridLayout(frame);
        gridLayout_4->setSpacing(0);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        GBColorMap = new QGroupBox(frame);
        GBColorMap->setObjectName(QString::fromUtf8("GBColorMap"));
        sizePolicy.setHeightForWidth(GBColorMap->sizePolicy().hasHeightForWidth());
        GBColorMap->setSizePolicy(sizePolicy);
        GBColorMap->setMinimumSize(QSize(0, 102));
        GBColorMap->setMaximumSize(QSize(16777215, 102));
        gridLayout_2 = new QGridLayout(GBColorMap);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label = new QLabel(GBColorMap);
        label->setObjectName(QString::fromUtf8("label"));
        label->setMinimumSize(QSize(40, 0));
        label->setMaximumSize(QSize(40, 16777215));

        horizontalLayout_3->addWidget(label);

        CB_Target = new QComboBox(GBColorMap);
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->addItem(QString());
        CB_Target->setObjectName(QString::fromUtf8("CB_Target"));

        horizontalLayout_3->addWidget(CB_Target);


        gridLayout_2->addLayout(horizontalLayout_3, 0, 0, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        RB_UserInput = new QRadioButton(GBColorMap);
        RB_UserInput->setObjectName(QString::fromUtf8("RB_UserInput"));

        horizontalLayout->addWidget(RB_UserInput);

        RB_FromResult = new QRadioButton(GBColorMap);
        RB_FromResult->setObjectName(QString::fromUtf8("RB_FromResult"));

        horizontalLayout->addWidget(RB_FromResult);

        PB_MapData = new QPushButton(GBColorMap);
        PB_MapData->setObjectName(QString::fromUtf8("PB_MapData"));
        PB_MapData->setMinimumSize(QSize(70, 0));
        PB_MapData->setMaximumSize(QSize(70, 16777215));

        horizontalLayout->addWidget(PB_MapData);


        gridLayout_2->addLayout(horizontalLayout, 1, 0, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        L_LimitMin = new QLabel(GBColorMap);
        L_LimitMin->setObjectName(QString::fromUtf8("L_LimitMin"));

        horizontalLayout_2->addWidget(L_LimitMin);

        LE_LimitMin = new QLineEdit(GBColorMap);
        LE_LimitMin->setObjectName(QString::fromUtf8("LE_LimitMin"));

        horizontalLayout_2->addWidget(LE_LimitMin);

        L_LimitMax = new QLabel(GBColorMap);
        L_LimitMax->setObjectName(QString::fromUtf8("L_LimitMax"));

        horizontalLayout_2->addWidget(L_LimitMax);

        LE_LimitMax = new QLineEdit(GBColorMap);
        LE_LimitMax->setObjectName(QString::fromUtf8("LE_LimitMax"));

        horizontalLayout_2->addWidget(LE_LimitMax);

        PB_Apply = new QPushButton(GBColorMap);
        PB_Apply->setObjectName(QString::fromUtf8("PB_Apply"));
        PB_Apply->setMinimumSize(QSize(50, 0));
        PB_Apply->setMaximumSize(QSize(50, 16777215));

        horizontalLayout_2->addWidget(PB_Apply);


        gridLayout_2->addLayout(horizontalLayout_2, 2, 0, 1, 1);


        verticalLayout->addWidget(GBColorMap);


        gridLayout_4->addLayout(verticalLayout, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wresult);

        QMetaObject::connectSlotsByName(wresult);
    } // setupUi

    void retranslateUi(QWidget *wresult)
    {
        wresult->setWindowTitle(QApplication::translate("wresult", "Form", nullptr));
        GBColorMap->setTitle(QApplication::translate("wresult", "ColorMap", nullptr));
        label->setText(QApplication::translate("wresult", "Type", nullptr));
        CB_Target->setItemText(0, QApplication::translate("wresult", "None", nullptr));
        CB_Target->setItemText(1, QApplication::translate("wresult", "Position Magnitude", nullptr));
        CB_Target->setItemText(2, QApplication::translate("wresult", "Position X", nullptr));
        CB_Target->setItemText(3, QApplication::translate("wresult", "Position Y", nullptr));
        CB_Target->setItemText(4, QApplication::translate("wresult", "Position Z", nullptr));
        CB_Target->setItemText(5, QApplication::translate("wresult", "Velocity Magnitude", nullptr));
        CB_Target->setItemText(6, QApplication::translate("wresult", "Velocity X", nullptr));
        CB_Target->setItemText(7, QApplication::translate("wresult", "Velocity Y", nullptr));
        CB_Target->setItemText(8, QApplication::translate("wresult", "Velocity Z", nullptr));

        RB_UserInput->setText(QApplication::translate("wresult", "User input", nullptr));
        RB_FromResult->setText(QApplication::translate("wresult", "From result", nullptr));
        PB_MapData->setText(QApplication::translate("wresult", "Map data", nullptr));
        L_LimitMin->setText(QApplication::translate("wresult", "Min", nullptr));
        L_LimitMax->setText(QApplication::translate("wresult", "Max", nullptr));
        PB_Apply->setText(QApplication::translate("wresult", "Apply", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wresult: public Ui_wresult {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WRESULT_H
