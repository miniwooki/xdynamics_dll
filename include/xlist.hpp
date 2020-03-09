#ifndef XLIST_H
#define XLIST_H

#include "xstring.h"
#include "xdynamics_decl.h"

template< class T>
class XDYNAMICS_API xlist
{
	struct xListNode
	{
		T value;
		xListNode* parent;
		xListNode* child;
		xListNode(T v, xListNode* n) : value(v), parent(n), child(nullptr) {}
	};
	xListNode* head;
	xListNode* tail;
	unsigned int sz;
public:
	xlist() : head(0), tail(0), sz(0) {}
	~xlist()
	{
		xListNode* n = head;
		while (n != nullptr) {
			xListNode *tn = n;
			n = n->child;
			delete tn;
			tn = nullptr;
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
		}
	}

	void push_front(T v)
	{
		xListNode *n = new xListNode(v, nullptr);
		if (head)
			head->parent = n;
		head = n;
		if (!sz) tail = head;
		sz++;
	}

	void push_back(T v)
	{
		if (!head)
		{
			tail = nullptr;
			head = new xListNode(v, nullptr);
			tail = head;
			sz++;
		}
		else
		{
			xListNode *n = new xListNode(v, tail);
			/*if (tail)
				n->right = tail;*/
			tail = n;
			sz++;
		}
	}

	class iterator
	{
		xListNode* current;
	public:
		typedef T value_type;
		iterator(xListNode* init = 0) : current(init) {}

		bool operator!=(const iterator& c)
		{
			return current != c.current_node();
		}
		void next()
		{
			current = current->child;
		}
		T value()
		{
			return current->value;
		}
		bool has_next()
		{
			return current;
		}

		xListNode* current_node() const { return current; }
	};
	xListNode* find_node(T k)
	{
		for (iterator i = this->begin(); i != this->end(); i.next())
		{
			if (k == i.value())
				return i.current_node();
		}
		return NULL;
	}
	iterator begin() { return iterator(head); }
	iterator end() { return iterator(0); }

};

#endif