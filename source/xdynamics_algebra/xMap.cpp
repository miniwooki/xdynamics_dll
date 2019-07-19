// #include "xdynamics_algebra/xMap.h"
// 
// const xMapDataBase xMapDataBase::shared_null = {-1, 0, { 0, 0 }, 0 };
// 
// xMapDataBase *xMapDataBase::createData()
// {
// 	xMapDataBase *d = new xMapDataBase;
// 
// 	d->ref = 1;
// 	d->size = 0;
// 	d->header.left = 0;
// 	d->header.right = 0;
// 	d->mostLeftNode = &(d->header);
// 
// 	return d;
// }
// 
// void xMapDataBase::freeData(xMapDataBase *d)
// {
// 	delete d;
// }
// 
// 
// void xMapDataBase::recalcMostLeftNode()
// {
// 	mostLeftNode = &header;
// 	while (mostLeftNode->left)
// 		mostLeftNode = mostLeftNode->left;
// }
// 
// xMapNodeBase *xMapDataBase::createNode(int alloc, int alignment, xMapNodeBase* parent, bool left)
// {
//  	xMapNodeBase *node = new xMapNodeBase;
// // 	//Q_CHECK_PTR(node);
//  	node->left = NULL;
//  	node->right = NULL;
// 	//memset(node, 0, alloc);
// 	++size;
// 
// 	if (parent) {
// 		if (left) {
// 			parent->left = node;
// 			if (parent == mostLeftNode)
// 				mostLeftNode = node;
// 		}
// 		else {
// 			parent->right = node;
// 		}
// 		node->setParent(parent);
// // 		rebalance(node);
// 	}
//  	return node;
// 	//return 0;
// }
