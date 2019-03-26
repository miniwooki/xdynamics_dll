// #include "xStringList.h"
// 
// xStringList::xStringList()
// 	: head(NULL)
// 	, tail(NULL)
// 	, temp(NULL)
// 	, curr(NULL)
// 	, sz(0)
// {
// 
// }
// 
// xStringList::xStringList(const xStringList& ls)
// 	: head(NULL)
// 	, tail(NULL)
// 	, temp(NULL)
// 	, curr(NULL)
// 	, sz(ls.size())
// {
// 	Node* h = ls.Head();
// 	for (unsigned int i = 0; i < sz; i++)
// 	{
// 		Node *nd = new Node;
// 		nd->data = h->data;
// 		nd->parent = NULL;
// 		nd->child = NULL;
// 		if (i)
// 		{
// 			tail->child = nd;
// 			nd->parent = tail;
// 			nd->child = NULL;
// 		}
// 		else
// 		{
// 			head = nd;
// 		}
// 		tail = nd;
// 		h = h->child;
// 	}
// }
// 
// xStringList::~xStringList()
// {
// 	if (tail->parent)
// 	{
// 		temp = tail->parent;
// 		while (temp->parent != NULL)
// 		{
// 			if (temp->child) delete temp->child; temp->child = NULL;
// 		}
// 	}
// }
// 
// xString& xStringList::operator++()
// {
// 	if (curr)
// 	{
// 		curr = curr->child;
// 		return curr->data;
// 	}
// 	return xString(L"");
// }
// 
// void xStringList::operator=(Node* node)
// {
// 	Node* _t = node;
// 	for (unsigned int i = 0; i < sz; i++)
// 	{
// 		Node *nd = new Node;
// 		nd->data = _t->data;
// 		nd->parent = NULL;
// 		nd->child = NULL;
// 		if (i)
// 		{
// 			tail->child = nd;
// 			nd->parent = tail;
// 			nd->child = NULL;
// 		}
// 		else
// 		{
// 			head = nd;
// 		}
// 		tail = nd;
// 		_t = _t->child;
// 	}
// }
// 
// // void xStringList::operator=(xStringList& ls)
// // {
// // 
// // }
// 
// // xString xStringList::at(int idx) const
// // {
// // 	//return xstr_list[idx];
// // }
// 
// unsigned int xStringList::size() const
// {
// 	return sz;
// }
// 
// void xStringList::push_back(xString xs)
// {
// 	Node *nd = new Node;
// 	nd->data = xs;
// 	nd->parent = NULL;
// 	nd->child = NULL;
// 	if (sz)
// 	{
// 		tail->child = nd;
// 		nd->parent = tail;
// 		nd->child = NULL;
// 	}
// 	else
// 	{
// 		head = nd;
// 	}
// 	tail = nd;
// 	sz++;
// }
// 
// xString& xStringList::begin()
// {
// 	curr = head;
// 	return curr->data;
// }
// 
// bool xStringList::IsEnd()
// {
// 	return curr ? false : true;
// }
// 
// xStringList::Node* xStringList::Head() const
// {
// 	return head;
// }
