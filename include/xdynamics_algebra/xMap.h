// #ifndef XMAP_H
// #define XMAP_H
// 
// #include "xdynamics_decl.h"
// 
// // template <class Key> inline bool xMapLessThanKey(const Key &key1, const Key &key2)
// // {
// // 	return key1 < key2;
// // }
// // 
// // template <class Ptr> inline bool xMapLessThanKey(const Ptr *key1, const Ptr *key2)
// // {
// // 	return std::less<const Ptr *>()(key1, key2);
// // }
// 
// struct XDYNAMICS_API xMapNodeBase
// {
// 	unsigned int p;
// 	xMapNodeBase* left;
// 	xMapNodeBase* right;
// 
// 	xMapNodeBase* parent() const { return reinterpret_cast<xMapNodeBase *>(p & ~3); }
// 	void setParent(xMapNodeBase *pp) { p = (p & 3) | (unsigned int)pp; }
// };
// 
// template <class Key, class T>
// struct xMapNode : public xMapNodeBase
// {
// 	Key key;
// 	T value;
// 
// 	inline xMapNode *leftNode() const { return static_cast<xMapNode *>(left); }
// 	inline xMapNode *rightNode() const { return static_cast<xMapNode *>(right); }
// 
// 	xMapNode<Key, T> *copy(xMapData<Key, T> *d) const;
// 	xMapNode<Key, T> *lowerBound(const Key &key);
// 	xMapNode<Key, T> *upperBound(const Key &key);
// };
// 
// template <class Key, class T>
// inline xMapNode<Key, T> *xMapNode<Key, T>::lowerBound(const Key &akey)
// {
// 	xMapNode<Key, T> *n = this;
// 	xMapNode<Key, T> *lastNode = nullptr;
// // 	while (n) {
// // 		if (!xMapLessThanKey(n->key, akey)) {
// // 			lastNode = n;
// // 			n = n->leftNode();
// // 		}
// // 		else {
// // 			n = n->rightNode();
// // 		}
// // 	}
// 	return lastNode;
// }
// 
// template <class Key, class T>
// inline xMapNode<Key, T> *xMapNode<Key, T>::upperBound(const Key &akey)
// {
// 	xMapNode<Key, T> *n = this;
// 	xMapNode<Key, T> *lastNode = nullptr;
// // 	while (n) {
// // 		if (xMapLessThanKey(akey, n->key)) {
// // 			lastNode = n;
// // 			n = n->leftNode();
// // 		}
// // 		else {
// // 			n = n->rightNode();
// // 		}
// // 	}
// 	return lastNode;
// }
// 
// template <class Key, class T>
// xMapNode<Key, T> *xMapNode<Key, T>::copy(xMapData<Key, T> *d) const
// {
// 	xMapNode<Key, T> *n = d->createNode(key, value);
// 	//n->setColor(color());
// 	if (left) {
// 		n->left = leftNode()->copy(d);
// 		n->left->setParent(n);
// 	}
// 	else {
// 		n->left = nullptr;
// 	}
// 	if (right) {
// 		n->right = rightNode()->copy(d);
// 		n->right->setParent(n);
// 	}
// 	else {
// 		n->right = nullptr;
// 	}
// 	return n;
// }
// 
// struct XDYNAMICS_API xMapDataBase
// {
// 	int ref;
// 	int size;
// 	xMapNodeBase header;
// 	xMapNodeBase *mostLeftNode;
// 	xMapNodeBase *createNode(int size, int alignment, xMapNodeBase* parent, bool left);
// 	static const xMapDataBase shared_null;
// 	void recalcMostLeftNode();
// 	static xMapDataBase *createData();
// 	static void freeData(xMapDataBase *d);
// };
// 
// template <class Key, class T>
// struct xMapData : public xMapDataBase
// {
// 	typedef xMapNode<Key, T> Node;
// 
// 	Node *root() const { return static_cast<Node *>(header.left); }
// 	const Node *end() const { return reinterpret_cast<const Node *>(&header); }
// 	Node *end() { return reinterpret_cast<Node *>(&header); }
// 	const Node *begin() const { if (root()) return static_cast<const Node*>(mostLeftNode); return end(); }
// 	Node *begin() { if (root()) return static_cast<Node*>(mostLeftNode); return end(); }
// 	Node *findNode(const Key &akey) const;
// 	Node *createNode(const Key &k, const T &v, Node *parent = nullptr, bool left = false)
// 	{
// 		Node* n = static_cast<Node *>(xMapDataBase::createNode(sizeof(Node), __alignof(Node), parent, left));
// 		new (&n->key) Key(k);
// 		new (&n->value) T(v);
// 		return n;
// 	}
// 	static xMapData *create(){
// 		return static_cast<xMapData *>(createData());
// 	}
// 	void destroy()
// 	{
// 		if (root())
// 		{
// 
// 		}
// 		freeData(this);
// 	}
// };
// 
// template <class Key, class T>
// xMapNode<Key, T> *xMapData<Key, T>::findNode(const Key &akey) const
// {
// 	Node *r = root();
// 	xMapNode<Key, T> *n = this;
// 	xMapNode<Key, T> *lastNode = nullptr;
// 	while (n)
// 	{
// 		if (n->key == akey)
// 			return n;
// 		else
// 			n = n->rightNode();
// 	}
//  		//Node *lb = r->lowerBound(akey);
// //  		if (lb && !xMapLessThanKey(akey, lb->key))
// //  			return lb;
// 	return nullptr;
// }
// 
// template < class Key, class T >
// class xMap
// {
// 	typedef xMapNode<Key, T> Node;
// 	xMapData<Key, T> *d;
// public:
// 	inline xMap() : dh(false), d(static_cast<xMapData<Key, T>*>(const_cast<xMapDataBase *>(&xMapDataBase::shared_null))) { }
// 	inline ~xMap() { if(d->ref != -1) d->destroy(); }
// 
// 	T &operator[](const Key &key);
// 	const T operator[](const Key &key) const;
// 
// 	class iterator
// 	{
// 		Node *i;
// 	public:
// 		inline iterator(Node *node) : i(node) {}
// 		inline T &operator*() const { return i->value; }
// 	};
// 
// 	iterator insert(const Key &key, const T &value);
// 	const T value(const Key& key) const;
// 
// private:
// 	void detach_helper();
// 	bool dh;
// };
// 
// template <class Key, class T>
// inline const T xMap<Key, T>::operator[](const Key &akey) const
// {
// 	return value(akey);
// }
// 
// template <class Key, class T>
// inline T &xMap<Key, T>::operator[](const Key &akey)
// {
// 	detach_helper();
// 	Node *n = d->findNode(akey);
// 	if (!n)
// 		return *insert(akey, T());
// 	return n->value;
// }
// 
// template <class Key, class T>
// inline typename xMap<Key, T>::iterator xMap<Key, T>::insert(const Key &akey, const T &avalue)
// {
// 	if (!dh) detach_helper();
// 	Node *n = d->root();
// 	Node *y = d->end();
// 	Node *lastNode = nullptr;
// 	bool left = true;
// 	Node *z = d->createNode(akey, avalue, y, left);
// 	return iterator(z);
// }
// 
// template<class Key, class T>
// inline void xMap<Key, T>::detach_helper()
// {
// 	if (d->ref == -1)
// 		return;
// 	xMapData<Key, T> *x = xMapData<Key, T>::create();
// 	if (d->header.left) {
// 		x->header.left = static_cast<Node *>(d->header.left)->copy(x);
// 		x->header.left->setParent(&x->header);
// 	}
// 	if (!d->ref == -1)
// 		d->destroy();
// 	d = x;
// 	d->recalcMostLeftNode();
// 	dh = true;
// }
// 
// template <class Key, class T>
// inline const T xMap<Key, T>::value(const Key &akey) const
// {
// 	Node *n = d->findNode(akey);
// 	return n ? n->value;
// }
// 
// #endif