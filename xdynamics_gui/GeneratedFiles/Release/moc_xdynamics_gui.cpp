/****************************************************************************
** Meta object code from reading C++ file 'xdynamics_gui.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.10.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../xdynamics_gui.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xdynamics_gui.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.10.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xdynamics_gui_t {
    QByteArrayData data[10];
    char stringdata0[126];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xdynamics_gui_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xdynamics_gui_t qt_meta_stringdata_xdynamics_gui = {
    {
QT_MOC_LITERAL(0, 0, 13), // "xdynamics_gui"
QT_MOC_LITERAL(1, 14, 4), // "xNew"
QT_MOC_LITERAL(2, 19, 0), // ""
QT_MOC_LITERAL(3, 20, 5), // "xSave"
QT_MOC_LITERAL(4, 26, 5), // "xOpen"
QT_MOC_LITERAL(5, 32, 20), // "xGetSimulationWidget"
QT_MOC_LITERAL(6, 53, 12), // "wsimulation*"
QT_MOC_LITERAL(7, 66, 20), // "xRunSimulationThread"
QT_MOC_LITERAL(8, 87, 21), // "xExitSimulationThread"
QT_MOC_LITERAL(9, 109, 16) // "xRecieveProgress"

    },
    "xdynamics_gui\0xNew\0\0xSave\0xOpen\0"
    "xGetSimulationWidget\0wsimulation*\0"
    "xRunSimulationThread\0xExitSimulationThread\0"
    "xRecieveProgress"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xdynamics_gui[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x08 /* Private */,
       3,    0,   50,    2, 0x08 /* Private */,
       4,    0,   51,    2, 0x08 /* Private */,
       5,    1,   52,    2, 0x08 /* Private */,
       7,    3,   55,    2, 0x08 /* Private */,
       8,    0,   62,    2, 0x08 /* Private */,
       9,    2,   63,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 6,    2,
    QMetaType::Void, QMetaType::Double, QMetaType::UInt, QMetaType::Double,    2,    2,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::QString,    2,    2,

       0        // eod
};

void xdynamics_gui::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        xdynamics_gui *_t = static_cast<xdynamics_gui *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->xNew(); break;
        case 1: _t->xSave(); break;
        case 2: _t->xOpen(); break;
        case 3: _t->xGetSimulationWidget((*reinterpret_cast< wsimulation*(*)>(_a[1]))); break;
        case 4: _t->xRunSimulationThread((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3]))); break;
        case 5: _t->xExitSimulationThread(); break;
        case 6: _t->xRecieveProgress((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObject xdynamics_gui::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_xdynamics_gui.data,
      qt_meta_data_xdynamics_gui,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *xdynamics_gui::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xdynamics_gui::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_xdynamics_gui.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int xdynamics_gui::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
