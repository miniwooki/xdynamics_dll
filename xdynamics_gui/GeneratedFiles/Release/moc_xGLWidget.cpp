/****************************************************************************
** Meta object code from reading C++ file 'xGLWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.10.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xGLWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xGLWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.10.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xGLWidget_t {
    QByteArrayData data[16];
    char stringdata0[213];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xGLWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xGLWidget_t qt_meta_stringdata_xGLWidget = {
    {
QT_MOC_LITERAL(0, 0, 9), // "xGLWidget"
QT_MOC_LITERAL(1, 10, 16), // "xRotationChanged"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 5), // "angle"
QT_MOC_LITERAL(4, 34, 16), // "yRotationChanged"
QT_MOC_LITERAL(5, 51, 16), // "zRotationChanged"
QT_MOC_LITERAL(6, 68, 21), // "changedAnimationFrame"
QT_MOC_LITERAL(7, 90, 23), // "signalGeometrySelection"
QT_MOC_LITERAL(8, 114, 16), // "releaseOperation"
QT_MOC_LITERAL(9, 131, 7), // "fitView"
QT_MOC_LITERAL(10, 139, 12), // "setXRotation"
QT_MOC_LITERAL(11, 152, 12), // "setYRotation"
QT_MOC_LITERAL(12, 165, 12), // "setZRotation"
QT_MOC_LITERAL(13, 178, 15), // "ShowContextMenu"
QT_MOC_LITERAL(14, 194, 3), // "pos"
QT_MOC_LITERAL(15, 198, 14) // "setSketchSpace"

    },
    "xGLWidget\0xRotationChanged\0\0angle\0"
    "yRotationChanged\0zRotationChanged\0"
    "changedAnimationFrame\0signalGeometrySelection\0"
    "releaseOperation\0fitView\0setXRotation\0"
    "setYRotation\0setZRotation\0ShowContextMenu\0"
    "pos\0setSketchSpace"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xGLWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       6,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   74,    2, 0x06 /* Public */,
       4,    1,   77,    2, 0x06 /* Public */,
       5,    1,   80,    2, 0x06 /* Public */,
       6,    0,   83,    2, 0x06 /* Public */,
       7,    1,   84,    2, 0x06 /* Public */,
       8,    0,   87,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       9,    0,   88,    2, 0x0a /* Public */,
      10,    1,   89,    2, 0x0a /* Public */,
      11,    1,   92,    2, 0x0a /* Public */,
      12,    1,   95,    2, 0x0a /* Public */,
      13,    1,   98,    2, 0x0a /* Public */,
      15,    0,  101,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::QPoint,   14,
    QMetaType::Void,

       0        // eod
};

void xGLWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        xGLWidget *_t = static_cast<xGLWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->xRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->yRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->zRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->changedAnimationFrame(); break;
        case 4: _t->signalGeometrySelection((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 5: _t->releaseOperation(); break;
        case 6: _t->fitView(); break;
        case 7: _t->setXRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->setYRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->setZRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->ShowContextMenu((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 11: _t->setSketchSpace(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            typedef void (xGLWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::xRotationChanged)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (xGLWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::yRotationChanged)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (xGLWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::zRotationChanged)) {
                *result = 2;
                return;
            }
        }
        {
            typedef void (xGLWidget::*_t)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::changedAnimationFrame)) {
                *result = 3;
                return;
            }
        }
        {
            typedef void (xGLWidget::*_t)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::signalGeometrySelection)) {
                *result = 4;
                return;
            }
        }
        {
            typedef void (xGLWidget::*_t)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::releaseOperation)) {
                *result = 5;
                return;
            }
        }
    }
}

const QMetaObject xGLWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_xGLWidget.data,
      qt_meta_data_xGLWidget,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *xGLWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xGLWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_xGLWidget.stringdata0))
        return static_cast<void*>(this);
    return QGLWidget::qt_metacast(_clname);
}

int xGLWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 12)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 12;
    }
    return _id;
}

// SIGNAL 0
void xGLWidget::xRotationChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void xGLWidget::yRotationChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void xGLWidget::zRotationChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void xGLWidget::changedAnimationFrame()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void xGLWidget::signalGeometrySelection(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void xGLWidget::releaseOperation()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE