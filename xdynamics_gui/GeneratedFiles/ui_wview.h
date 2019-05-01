/********************************************************************************
** Form generated from reading UI file 'wviewOJeFJO.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef WVIEWOJEFJO_H
#define WVIEWOJEFJO_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wview
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QGroupBox *GBTransparency;
    QGridLayout *gridLayout_4;
    QSlider *HSTransparency;
    QLineEdit *LETransparency;
    QGroupBox *GBColor;
    QGridLayout *gridLayout_3;
    QLabel *label;
    QSlider *HSRed;
    QLineEdit *LERed;
    QLabel *label_2;
    QSlider *HSGreen;
    QLineEdit *LEGreen;
    QLabel *label_3;
    QSlider *HSBlue;
    QLineEdit *LEBlue;
    QPushButton *PBPalette;

    void setupUi(QWidget *wview)
    {
        if (wview->objectName().isEmpty())
            wview->setObjectName(QStringLiteral("wview"));
        wview->resize(230, 200);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wview->sizePolicy().hasHeightForWidth());
        wview->setSizePolicy(sizePolicy);
        wview->setMinimumSize(QSize(230, 200));
        wview->setMaximumSize(QSize(230, 200));
        gridLayout = new QGridLayout(wview);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wview);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setSpacing(0);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        GBTransparency = new QGroupBox(frame);
        GBTransparency->setObjectName(QStringLiteral("GBTransparency"));
        gridLayout_4 = new QGridLayout(GBTransparency);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        HSTransparency = new QSlider(GBTransparency);
        HSTransparency->setObjectName(QStringLiteral("HSTransparency"));
        HSTransparency->setOrientation(Qt::Horizontal);
        HSTransparency->setTickPosition(QSlider::TicksBelow);

        gridLayout_4->addWidget(HSTransparency, 0, 0, 1, 1);

        LETransparency = new QLineEdit(GBTransparency);
        LETransparency->setObjectName(QStringLiteral("LETransparency"));
        LETransparency->setMinimumSize(QSize(40, 20));
        LETransparency->setMaximumSize(QSize(40, 20));

        gridLayout_4->addWidget(LETransparency, 0, 1, 1, 1);


        gridLayout_2->addWidget(GBTransparency, 0, 0, 1, 1);

        GBColor = new QGroupBox(frame);
        GBColor->setObjectName(QStringLiteral("GBColor"));
        gridLayout_3 = new QGridLayout(GBColor);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setContentsMargins(5, 5, 5, 5);
        label = new QLabel(GBColor);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_3->addWidget(label, 0, 0, 1, 1);

        HSRed = new QSlider(GBColor);
        HSRed->setObjectName(QStringLiteral("HSRed"));
        HSRed->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(HSRed, 0, 1, 1, 1);

        LERed = new QLineEdit(GBColor);
        LERed->setObjectName(QStringLiteral("LERed"));
        LERed->setMinimumSize(QSize(40, 20));
        LERed->setMaximumSize(QSize(40, 20));

        gridLayout_3->addWidget(LERed, 0, 2, 1, 1);

        label_2 = new QLabel(GBColor);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_3->addWidget(label_2, 1, 0, 1, 1);

        HSGreen = new QSlider(GBColor);
        HSGreen->setObjectName(QStringLiteral("HSGreen"));
        HSGreen->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(HSGreen, 1, 1, 1, 1);

        LEGreen = new QLineEdit(GBColor);
        LEGreen->setObjectName(QStringLiteral("LEGreen"));
        LEGreen->setMinimumSize(QSize(40, 20));
        LEGreen->setMaximumSize(QSize(40, 20));

        gridLayout_3->addWidget(LEGreen, 1, 2, 1, 1);

        label_3 = new QLabel(GBColor);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_3->addWidget(label_3, 2, 0, 1, 1);

        HSBlue = new QSlider(GBColor);
        HSBlue->setObjectName(QStringLiteral("HSBlue"));
        HSBlue->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(HSBlue, 2, 1, 1, 1);

        LEBlue = new QLineEdit(GBColor);
        LEBlue->setObjectName(QStringLiteral("LEBlue"));
        LEBlue->setMinimumSize(QSize(40, 20));
        LEBlue->setMaximumSize(QSize(40, 20));

        gridLayout_3->addWidget(LEBlue, 2, 2, 1, 1);

        PBPalette = new QPushButton(GBColor);
        PBPalette->setObjectName(QStringLiteral("PBPalette"));
        PBPalette->setMinimumSize(QSize(0, 28));
        PBPalette->setMaximumSize(QSize(16777215, 28));

        gridLayout_3->addWidget(PBPalette, 3, 0, 1, 3);


        gridLayout_2->addWidget(GBColor, 1, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wview);

        QMetaObject::connectSlotsByName(wview);
    } // setupUi

    void retranslateUi(QWidget *wview)
    {
        wview->setWindowTitle(QApplication::translate("wview", "Form", nullptr));
        GBTransparency->setTitle(QApplication::translate("wview", "Transparency", nullptr));
        GBColor->setTitle(QApplication::translate("wview", "Color", nullptr));
        label->setText(QApplication::translate("wview", "Red", nullptr));
        label_2->setText(QApplication::translate("wview", "Green", nullptr));
        label_3->setText(QApplication::translate("wview", "Blue", nullptr));
        PBPalette->setText(QApplication::translate("wview", "Palette", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wview: public Ui_wview {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WVIEWOJEFJO_H
