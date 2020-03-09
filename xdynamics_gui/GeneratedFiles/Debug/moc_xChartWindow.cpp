/****************************************************************************
** Meta object code from reading C++ file 'xChartWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xChartWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xChartWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xChartWindow_t {
    QByteArrayData data[7];
    char stringdata0[96];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xChartWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xChartWindow_t qt_meta_stringdata_xChartWindow = {
    {
QT_MOC_LITERAL(0, 0, 12), // "xChartWindow"
QT_MOC_LITERAL(1, 13, 16), // "updateTargetItem"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 26), // "click_passing_distribution"
QT_MOC_LITERAL(4, 58, 20), // "PlotFromComboBoxItem"
QT_MOC_LITERAL(5, 79, 1), // "i"
QT_MOC_LITERAL(6, 81, 14) // "editingCommand"

    },
    "xChartWindow\0updateTargetItem\0\0"
    "click_passing_distribution\0"
    "PlotFromComboBoxItem\0i\0editingCommand"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xChartWindow[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    3,   39,    2, 0x0a /* Public */,
       3,    0,   46,    2, 0x08 /* Private */,
       4,    1,   47,    2, 0x08 /* Private */,
       4,    0,   50,    2, 0x28 /* Private | MethodCloned */,
       6,    0,   51,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::QString, QMetaType::QStringList,    2,    2,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void xChartWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<xChartWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateTargetItem((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QStringList(*)>(_a[3]))); break;
        case 1: _t->click_passing_distribution(); break;
        case 2: _t->PlotFromComboBoxItem((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->PlotFromComboBoxItem(); break;
        case 4: _t->editingCommand(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject xChartWindow::staticMetaObject = { {
    &QMainWindow::staticMetaObject,
    qt_meta_stringdata_xChartWindow.data,
    qt_meta_data_xChartWindow,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *xChartWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xChartWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_xChartWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int xChartWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
