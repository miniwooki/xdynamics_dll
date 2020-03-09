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
		xMapNode* right;
		xMapNode* left;
		xMapNode(Key k, T v) : key(k), value(v), left(nullptr), right(nullptr) {}
	};
	xMapNode* head;
	xMapNode* tail;
	unsigned int sz;
public:
	xmap() : head(0), tail(0), sz(0) {}
	~xmap()
	{
		xMapNode *n = head;
		while (n != nullptr)
		{
			xMapNode *tn = n;
			n = n->left;
			delete tn;
			tn = NULL;
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

	void push_front(Key k, T v)
	{
		xMapNode *n = new xMapNode(k, v);
		n->left = head;
		if (head)
			head->right = n;
		head = n;
		//head->right = NULL;
		if (!sz) head->left = tail;
		sz++;
	}

	T operator[] (Key k)
	{
		xMapNode* n = find_node(k);
		if (!n)
		{
			return NULL;// head->value;
		}
		return n->value;
	}

	int insert(Key k, T v)
	{
		if (!sz)
		{
			xMapNode *n = new xMapNode(k, v);
			head = n;
			head->right = NULL;
			tail = NULL;
			head->left = tail;
			sz++;
		}
		else
		{
			xMapNode* n = find_node(k);
			if (!n)
			{
				xMapNode *n = new xMapNode(k, v);
				iterator it = begin();
				for (; it != end(); it.next())
				{
					if (it.key() > n->key)
						break;
				}
				xMapNode *cn = it.current_node();
				if (cn == head)
				{
					n->left = head;
					head->right = n;
					if (!tail)
						tail = head;
					head = n;

				}
				else if (it == end())
				{
					if (!tail)
					{
						tail = n;
						n->right = head;
						head->left = n;
					}
					else
					{
						n->right = tail;
						tail->left = n;
						tail = n;
					}					
				}
				else
				{
					xMapNode *cn_rnode = cn->right;
					n->left = cn;
					n->right = cn_rnode;
					cn_rnode->left = n;
					cn->right = n;
				}
				sz++;
			}
			return 0;
		}
		return -1;
	}

	T take(Key k)
	{
		xMapNode* n = find_node(k);
		if (n)
		{
			xMapNode *nl = n->left;
			xMapNode *nr = n->right;
			if(nl)
				if (nr)
					nr->left = nl;
			sz--;
			T o = n->value;
			delete n;
			n = NULL;
			if (sz == 0)
				head = nullptr;

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
		void setValue(T v)
		{
			current->value = v;
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