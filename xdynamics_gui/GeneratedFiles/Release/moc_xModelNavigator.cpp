/****************************************************************************
** Meta object code from reading C++ file 'xModelNavigator.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.10.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xModelNavigator.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xModelNavigator.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.10.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_wview_t {
    QByteArrayData data[7];
    char stringdata0[87];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_wview_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_wview_t qt_meta_stringdata_wview = {
    {
QT_MOC_LITERAL(0, 0, 5), // "wview"
QT_MOC_LITERAL(1, 6, 18), // "changeTransparency"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 12), // "colorPalette"
QT_MOC_LITERAL(4, 39, 14), // "changeRedColor"
QT_MOC_LITERAL(5, 54, 16), // "changeGreenColor"
QT_MOC_LITERAL(6, 71, 15) // "changeBlueColor"

    },
    "wview\0changeTransparency\0\0colorPalette\0"
    "changeRedColor\0changeGreenColor\0"
    "changeBlueColor"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_wview[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x08 /* Private */,
       3,    0,   42,    2, 0x08 /* Private */,
       4,    1,   43,    2, 0x08 /* Private */,
       5,    1,   46,    2, 0x08 /* Private */,
       6,    1,   49,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Int,    2,

       0        // eod
};

void wview::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        wview *_t = static_cast<wview *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->changeTransparency((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->colorPalette(); break;
        case 2: _t->changeRedColor((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->changeGreenColor((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->changeBlueColor((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject wview::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_wview.data,
      qt_meta_data_wview,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *wview::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *wview::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_wview.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "Ui::wview"))
        return static_cast< Ui::wview*>(this);
    return QWidget::qt_metacast(_clname);
}

int wview::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
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
struct qt_meta_stringdata_xModelNavigator_t {
    QByteArrayData data[6];
    char stringdata0[83];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xModelNavigator_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xModelNavigator_t qt_meta_stringdata_xModelNavigator = {
    {
QT_MOC_LITERAL(0, 0, 15), // "xModelNavigator"
QT_MOC_LITERAL(1, 16, 23), // "definedSimulationWidget"
QT_MOC_LITERAL(2, 40, 0), // ""
QT_MOC_LITERAL(3, 41, 12), // "wsimulation*"
QT_MOC_LITERAL(4, 54, 11), // "clickAction"
QT_MOC_LITERAL(5, 66, 16) // "QTreeWidgetItem*"

    },
    "xModelNavigator\0definedSimulationWidget\0"
    "\0wsimulation*\0clickAction\0QTreeWidgetItem*"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xModelNavigator[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   24,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    2,   27,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    2,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 5, QMetaType::Int,    2,    2,

       0        // eod
};

void xModelNavigator::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        xModelNavigator *_t = static_cast<xModelNavigator *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->definedSimulationWidget((*reinterpret_cast< wsimulation*(*)>(_a[1]))); break;
        case 1: _t->clickAction((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< wsimulation* >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            typedef void (xModelNavigator::*_t)(wsimulation * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xModelNavigator::definedSimulationWidget)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject xModelNavigator::staticMetaObject = {
    { &QDockWidget::staticMetaObject, qt_meta_stringdata_xModelNavigator.data,
      qt_meta_data_xModelNavigator,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *xModelNavigator::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xModelNavigator::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_xModelNavigator.stringdata0))
        return static_cast<void*>(this);
    return QDockWidget::qt_metacast(_clname);
}

int xModelNavigator::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDockWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void xModelNavigator::definedSimulationWidget(wsimulation * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
