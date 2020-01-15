/****************************************************************************
** Meta object code from reading C++ file 'xGLWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xGLWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xGLWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xGLWidget_t {
    QByteArrayData data[20];
    char stringdata0[282];
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
QT_MOC_LITERAL(9, 131, 13), // "contextSignal"
QT_MOC_LITERAL(10, 145, 15), // "contextMenuType"
QT_MOC_LITERAL(11, 161, 7), // "fitView"
QT_MOC_LITERAL(12, 169, 12), // "setXRotation"
QT_MOC_LITERAL(13, 182, 12), // "setYRotation"
QT_MOC_LITERAL(14, 195, 12), // "setZRotation"
QT_MOC_LITERAL(15, 208, 15), // "ShowContextMenu"
QT_MOC_LITERAL(16, 224, 3), // "pos"
QT_MOC_LITERAL(17, 228, 14), // "setSketchSpace"
QT_MOC_LITERAL(18, 243, 36), // "setupParticleBufferColorDistr..."
QT_MOC_LITERAL(19, 280, 1) // "n"

    },
    "xGLWidget\0xRotationChanged\0\0angle\0"
    "yRotationChanged\0zRotationChanged\0"
    "changedAnimationFrame\0signalGeometrySelection\0"
    "releaseOperation\0contextSignal\0"
    "contextMenuType\0fitView\0setXRotation\0"
    "setYRotation\0setZRotation\0ShowContextMenu\0"
    "pos\0setSketchSpace\0"
    "setupParticleBufferColorDistribution\0"
    "n"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xGLWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      15,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       7,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   89,    2, 0x06 /* Public */,
       4,    1,   92,    2, 0x06 /* Public */,
       5,    1,   95,    2, 0x06 /* Public */,
       6,    0,   98,    2, 0x06 /* Public */,
       7,    1,   99,    2, 0x06 /* Public */,
       8,    0,  102,    2, 0x06 /* Public */,
       9,    2,  103,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      11,    0,  108,    2, 0x0a /* Public */,
      12,    1,  109,    2, 0x0a /* Public */,
      13,    1,  112,    2, 0x0a /* Public */,
      14,    1,  115,    2, 0x0a /* Public */,
      15,    1,  118,    2, 0x0a /* Public */,
      17,    0,  121,    2, 0x0a /* Public */,
      18,    1,  122,    2, 0x0a /* Public */,
      18,    0,  125,    2, 0x2a /* Public | MethodCloned */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString, 0x80000000 | 10,    2,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::QPoint,   16,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   19,
    QMetaType::Void,

       0        // eod
};

void xGLWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<xGLWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->xRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->yRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->zRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->changedAnimationFrame(); break;
        case 4: _t->signalGeometrySelection((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 5: _t->releaseOperation(); break;
        case 6: _t->contextSignal((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< contextMenuType(*)>(_a[2]))); break;
        case 7: _t->fitView(); break;
        case 8: _t->setXRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->setYRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->setZRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->ShowContextMenu((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 12: _t->setSketchSpace(); break;
        case 13: _t->setupParticleBufferColorDistribution((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: _t->setupParticleBufferColorDistribution(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (xGLWidget::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::xRotationChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (xGLWidget::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::yRotationChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (xGLWidget::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::zRotationChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (xGLWidget::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::changedAnimationFrame)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (xGLWidget::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::signalGeometrySelection)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (xGLWidget::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::releaseOperation)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (xGLWidget::*)(QString , contextMenuType );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&xGLWidget::contextSignal)) {
                *result = 6;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject xGLWidget::staticMetaObject = { {
    &QGLWidget::staticMetaObject,
    qt_meta_stringdata_xGLWidget.data,
    qt_meta_data_xGLWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


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
        if (_id < 15)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 15;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 15)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 15;
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

// SIGNAL 6
void xGLWidget::contextSignal(QString _t1, contextMenuType _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
