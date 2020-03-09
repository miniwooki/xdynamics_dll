/****************************************************************************
** Meta object code from reading C++ file 'xSimulationWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xSimulationWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xSimulationWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_wsimulation_t {
    QByteArrayData data[10];
    char stringdata0[154];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_wsimulation_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_wsimulation_t qt_meta_stringdata_wsimulation = {
    {
QT_MOC_LITERAL(0, 0, 11), // "wsimulation"
QT_MOC_LITERAL(1, 12, 18), // "clickedSolveButton"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 23), // "clickedStartPointButton"
QT_MOC_LITERAL(4, 56, 17), // "clickedStopButton"
QT_MOC_LITERAL(5, 74, 17), // "UpdateInformation"
QT_MOC_LITERAL(6, 92, 11), // "SolveButton"
QT_MOC_LITERAL(7, 104, 10), // "StopButton"
QT_MOC_LITERAL(8, 115, 19), // "StartingPointButton"
QT_MOC_LITERAL(9, 135, 18) // "CheckStartingPoint"

    },
    "wsimulation\0clickedSolveButton\0\0"
    "clickedStartPointButton\0clickedStopButton\0"
    "UpdateInformation\0SolveButton\0StopButton\0"
    "StartingPointButton\0CheckStartingPoint"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_wsimulation[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    3,   54,    2, 0x06 /* Public */,
       3,    0,   61,    2, 0x06 /* Public */,
       4,    0,   62,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,   63,    2, 0x0a /* Public */,
       6,    0,   64,    2, 0x08 /* Private */,
       7,    0,   65,    2, 0x08 /* Private */,
       8,    0,   66,    2, 0x08 /* Private */,
       9,    1,   67,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double, QMetaType::UInt, QMetaType::Double,    2,    2,    2,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,

       0        // eod
};

void wsimulation::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<wsimulation *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->clickedSolveButton((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3]))); break;
        case 1: _t->clickedStartPointButton(); break;
        case 2: _t->clickedStopButton(); break;
        case 3: _t->UpdateInformation(); break;
        case 4: _t->SolveButton(); break;
        case 5: _t->StopButton(); break;
        case 6: _t->StartingPointButton(); break;
        case 7: _t->CheckStartingPoint((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (wsimulation::*)(double , unsigned int , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&wsimulation::clickedSolveButton)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (wsimulation::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&wsimulation::clickedStartPointButton)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (wsimulation::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&wsimulation::clickedStopButton)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject wsimulation::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_wsimulation.data,
    qt_meta_data_wsimulation,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *wsimulation::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *wsimulation::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_wsimulation.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "Ui::wsimulation"))
        return static_cast< Ui::wsimulation*>(this);
    return QWidget::qt_metacast(_clname);
}

int wsimulation::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void wsimulation::clickedSolveButton(double _t1, unsigned int _t2, double _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void wsimulation::clickedStartPointButton()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void wsimulation::clickedStopButton()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
