// #ifndef XSTRINGLIST_H
// #define XSTRINGLIST_H
// 
// #include "xdynamics_decl.h"
// #include "xString.h"
// 
// class XDYNAMICS_API xStringList
// {
// 	struct Node
// 	{
// 		xString data;
// 		Node *parent;
// 		Node *child;
// 	};
// public:
// 	xStringList();
// 	xStringList(const xStringList& ls);
// 	~xStringList();
// 
// 	xString& operator++ ();
// 	void operator= (xStringList& ls);
// 	xString at(int idx) const;
// 	unsigned int size() const;
// 	void push_back(xString xs);
// 	xString& begin();
// 	bool IsEnd();
// 	Node* Head() const;
// 
// private:
// 	unsigned int sz;
// 	//xString* xstr_list;
// 	Node* curr;
// 	Node* head;
// 	Node* tail;
// 	Node* temp;
// };
// 
// #endif