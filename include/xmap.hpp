#ifndef XMAP_HPP
#define XMAP_HPP

#include "xdynamics_decl.h"

template< class Key, class T>
class XDYNAMICS_API xmap
{
	struct xMapNode
	{
		Key key;
		T value;
		xMapNode* left;
		xMapNode* right;
		xMapNode(Key k, T v, xMapNode* n) : key(k), value(v), left(n) {}
	};
	xMapNode* head;
	xMapNode* tail;
	unsigned int sz;
public:
	xmap() : head(0), tail(0), sz(0) {}
	~xmap()
	{
		xMapNode *n = tail;
		while (n != head)
		{
			xMapNode *tn = n;
			//n = n->right;
			delete tn;
			tn = NULL;
			sz--;
		}
		if (head)
		{
			delete head;
			head = NULL;
			sz--;
		}
	}

	unsigned int size() { return sz; }

	void delete_all()
	{
		iterator it = begin();
		for (; it != end(); it.next())
		{
			delete it.value();
			//delete it.current_node();
		//	sz--;
		}
	}

	void push_front(Key k, T v)
	{
		/*if(head)
			head->right = head;*/
		xMapNode *n = new xMapNode(k, v, head);
		if (head)
			head->right = n;
		head = n;
		head->right = NULL;
		//	head->right = head;
		if (!sz) tail = head;
		sz++;
	}

	T operator[] (Key k)
	{
		xMapNode* n = find_node(k);
		if (!n)
		{
			//push_front(k, T());
			return NULL;// head->value;
		}
		return n->value;
	}

	int insert(Key k, T v)
	{
		xMapNode* n = find_node(k);
		if (!n)
		{
			push_front(k, v);
			return 0;
			//return NULL;// head->value;
		}
		return -1;
	}

	T take(Key k)
	{
		xMapNode* n = find_node(k);
		if (n)
		{
			if (n != head)
				n->right->left = n->left;
			else
				head = n->left;
			if (n != tail)
				n->left->right = n->right;
			else
				tail = n->right;
			sz--;
			T o = n->value;
			delete n;
			n = NULL;
			return o;
		}
		return NULL;
	}

	void erase(Key k)
	{
		xMapNode* n = find_node(k);
		if (n)
		{
			if (n != head)
				n->right->left = n->left;
			else
				head = n->left;
			if (n != tail)
				n->left->right = n->right;
			else
				tail = n->right;
			sz--;
			delete n;
		}
	}

	class iterator
	{
		xMapNode* current;
	public:
		typedef T value_type;
		iterator(xMapNode* init = 0) : current(init) {}

		bool operator!=(const iterator& c)
		{
			return current != c.current_node();
		}
		bool operator==(const iterator& c)
		{
			return current == c.current_node();
		}
		void next()
		{
			current = current->left;
		}
		Key& key()
		{
			return current->key;
		}
		T value()
		{
			return current->value;
		}
		bool has_next()
		{
			return current;
		}

		xMapNode* current_node() const { return current; }
	};
	iterator find(Key k)
	{
		for (iterator i = this->begin(); i != this->end(); i.next())
		{
			if (k == i.key())
				return i;
		}
		return iterator(0);
	}
	xMapNode* find_node(Key k)
	{
		for (iterator i = this->begin(); i != this->end(); i.next())
		{
			if (k == i.key())
				return i.current_node();
		}
		return NULL;
	}
	iterator begin() { return iterator(head); }
	iterator end() { return iterator(0); }
};

#endif