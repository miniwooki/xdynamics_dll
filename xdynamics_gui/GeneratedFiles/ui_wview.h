/********************************************************************************
** Form generated from reading UI file 'wview.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WVIEW_H
#define UI_WVIEW_H

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
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wview
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_3;
    QVBoxLayout *verticalLayout;
    QGroupBox *GBTransparency;
    QGridLayout *gridLayout_4;
    QSlider *HSTransparency;
    QLineEdit *LETransparency;
    QGroupBox *GBColor;
    QGridLayout *gridLayout_5;
    QVBoxLayout *verticalLayout_2;
    QGridLayout *gridLayout_2;
    QLabel *LRed;
    QSlider *HSRed;
    QLineEdit *LERed;
    QLabel *LGreen;
    QSlider *HSGreen;
    QLineEdit *LEGreen;
    QLabel *LBlue;
    QSlider *HSBlue;
    QLineEdit *LEBlue;
    QPushButton *PBPalette;

    void setupUi(QWidget *wview)
    {
        if (wview->objectName().isEmpty())
            wview->setObjectName(QStringLiteral("wview"));
        wview->resize(268, 220);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wview->sizePolicy().hasHeightForWidth());
        wview->setSizePolicy(sizePolicy);
        wview->setMinimumSize(QSize(0, 220));
        wview->setMaximumSize(QSize(16777215, 220));
        gridLayout = new QGridLayout(wview);
        gridLayout->setSpacing(5);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wview);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setMinimumSize(QSize(200, 0));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_3 = new QGridLayout(frame);
        gridLayout_3->setSpacing(0);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        GBTransparency = new QGroupBox(frame);
        GBTransparency->setObjectName(QStringLiteral("GBTransparency"));
        sizePolicy.setHeightForWidth(GBTransparency->sizePolicy().hasHeightForWidth());
        GBTransparency->setSizePolicy(sizePolicy);
        GBTransparency->setMinimumSize(QSize(0, 52));
        gridLayout_4 = new QGridLayout(GBTransparency);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(5, 5, 5, 5);
        HSTransparency = new QSlider(GBTransparency);
        HSTransparency->setObjectName(QStringLiteral("HSTransparency"));
        HSTransparency->setMaximum(100);
        HSTransparency->setOrientation(Qt::Horizontal);
        HSTransparency->setInvertedAppearance(false);
        HSTransparency->setInvertedControls(false);
        HSTransparency->setTickPosition(QSlider::TicksBelow);
        HSTransparency->setTickInterval(10);

        gridLayout_4->addWidget(HSTransparency, 0, 0, 1, 1);

        LETransparency = new QLineEdit(GBTransparency);
        LETransparency->setObjectName(QStringLiteral("LETransparency"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(LETransparency->sizePolicy().hasHeightForWidth());
        LETransparency->setSizePolicy(sizePolicy1);
        LETransparency->setMinimumSize(QSize(40, 20));
        LETransparency->setMaximumSize(QSize(40, 20));

        gridLayout_4->addWidget(LETransparency, 0, 1, 1, 1);


        verticalLayout->addWidget(GBTransparency);

        GBColor = new QGroupBox(frame);
        GBColor->setObjectName(QStringLiteral("GBColor"));
        sizePolicy.setHeightForWidth(GBColor->sizePolicy().hasHeightForWidth());
        GBColor->setSizePolicy(sizePolicy);
        GBColor->setMinimumSize(QSize(0, 140));
        gridLayout_5 = new QGridLayout(GBColor);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setHorizontalSpacing(6);
        LRed = new QLabel(GBColor);
        LRed->setObjectName(QStringLiteral("LRed"));

        gridLayout_2->addWidget(LRed, 0, 0, 1, 1);

        HSRed = new QSlider(GBColor);
        HSRed->setObjectName(QStringLiteral("HSRed"));
        HSRed->setMaximum(255);
        HSRed->setOrientation(Qt::Horizontal);
        HSRed->setInvertedAppearance(false);
        HSRed->setInvertedControls(false);
        HSRed->setTickPosition(QSlider::NoTicks);
        HSRed->setTickInterval(10);

        gridLayout_2->addWidget(HSRed, 0, 1, 1, 1);

        LERed = new QLineEdit(GBColor);
        LERed->setObjectName(QStringLiteral("LERed"));
        sizePolicy1.setHeightForWidth(LERed->sizePolicy().hasHeightForWidth());
        LERed->setSizePolicy(sizePolicy1);
        LERed->setMinimumSize(QSize(40, 20));
        LERed->setMaximumSize(QSize(40, 20));

        gridLayout_2->addWidget(LERed, 0, 2, 1, 1);

        LGreen = new QLabel(GBColor);
        LGreen->setObjectName(QStringLiteral("LGreen"));

        gridLayout_2->addWidget(LGreen, 1, 0, 1, 1);

        HSGreen = new QSlider(GBColor);
        HSGreen->setObjectName(QStringLiteral("HSGreen"));
        HSGreen->setMaximum(255);
        HSGreen->setOrientation(Qt::Horizontal);
        HSGreen->setInvertedAppearance(false);
        HSGreen->setInvertedControls(false);
        HSGreen->setTickPosition(QSlider::NoTicks);
        HSGreen->setTickInterval(10);

        gridLayout_2->addWidget(HSGreen, 1, 1, 1, 1);

        LEGreen = new QLineEdit(GBColor);
        LEGreen->setObjectName(QStringLiteral("LEGreen"));
        sizePolicy1.setHeightForWidth(LEGreen->sizePolicy().hasHeightForWidth());
        LEGreen->setSizePolicy(sizePolicy1);
        LEGreen->setMinimumSize(QSize(40, 20));
        LEGreen->setMaximumSize(QSize(40, 20));

        gridLayout_2->addWidget(LEGreen, 1, 2, 1, 1);

        LBlue = new QLabel(GBColor);
        LBlue->setObjectName(QStringLiteral("LBlue"));

        gridLayout_2->addWidget(LBlue, 2, 0, 1, 1);

        HSBlue = new QSlider(GBColor);
        HSBlue->setObjectName(QStringLiteral("HSBlue"));
        HSBlue->setMaximum(255);
        HSBlue->setOrientation(Qt::Horizontal);
        HSBlue->setInvertedAppearance(false);
        HSBlue->setInvertedControls(false);
        HSBlue->setTickPosition(QSlider::NoTicks);
        HSBlue->setTickInterval(10);

        gridLayout_2->addWidget(HSBlue, 2, 1, 1, 1);

        LEBlue = new QLineEdit(GBColor);
        LEBlue->setObjectName(QStringLiteral("LEBlue"));
        sizePolicy1.setHeightForWidth(LEBlue->sizePolicy().hasHeightForWidth());
        LEBlue->setSizePolicy(sizePolicy1);
        LEBlue->setMinimumSize(QSize(40, 20));
        LEBlue->setMaximumSize(QSize(40, 20));

        gridLayout_2->addWidget(LEBlue, 2, 2, 1, 1);


        verticalLayout_2->addLayout(gridLayout_2);

        PBPalette = new QPushButton(GBColor);
        PBPalette->setObjectName(QStringLiteral("PBPalette"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(PBPalette->sizePolicy().hasHeightForWidth());
        PBPalette->setSizePolicy(sizePolicy2);

        verticalLayout_2->addWidget(PBPalette);


        gridLayout_5->addLayout(verticalLayout_2, 0, 0, 1, 1);


        verticalLayout->addWidget(GBColor);


        gridLayout_3->addLayout(verticalLayout, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wview);

        QMetaObject::connectSlotsByName(wview);
    } // setupUi

    void retranslateUi(QWidget *wview)
    {
        wview->setWindowTitle(QApplication::translate("wview", "Form", nullptr));
        GBTransparency->setTitle(QApplication::translate("wview", "Transparency", nullptr));
        GBColor->setTitle(QApplication::translate("wview", "Color", nullptr));
        LRed->setText(QApplication::translate("wview", "Red", nullptr));
        LGreen->setText(QApplication::translate("wview", "Green", nullptr));
        LBlue->setText(QApplication::translate("wview", "Blue", nullptr));
        PBPalette->setText(QApplication::translate("wview", "Palette", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wview: public Ui_wview {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WVIEW_H
