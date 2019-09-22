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
		xListNode* left;
		xListNode* right;
		xListNode(T v, xListNode* n) : value(v), left(n) {}
	};
	xListNode* head;
	xListNode* tail;
	unsigned int sz;
public:
	xlist() : head(0), sz(0) {}
	~xlist()
	{
		remove_all();
	}

	unsigned int size() { return sz; }

	void remove_all()
	{
		xListNode *n = tail;
		while (n != head)
		{
			xListNode *tn = n;
			n = n->right;
			delete tn;
			tn = NULL;
		}
		if (head)
		{
			delete head;
			head = NULL;
		}
		sz = 0;
	}

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
		xListNode *n = new xListNode(v, head);
		if (head)
			head->right = n;
		head = n;
		head->right = NULL;
		if (!sz) tail = head;
		sz++;
	}

	void push_back(T v)
	{
		if (!head)
		{
			tail = NULL;
			head = new xListNode(v, tail);			
			tail = head;
			sz++;
		}
		else
		{
			xListNode *n = new xListNode(v, NULL);
			if (tail)
				n->right = tail;
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
			current = current->left;
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