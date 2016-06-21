

class tNode(object):
    def __init__(self,idx=-1,word=None):
        self.left = None
        self.right = None
        self.word = word
        self.size = 0
        self.height = 1
        self.parent = None
        self.label = None
        self.children = []
        self.idx=idx
        self.span=None

    def add_parent(self,parent):
        self.parent=parent
    def add_child(self,node):
        assert len(self.children) < 2
        self.children.append(node)
    def add_children(self,children):
        self.children.extend(children)

    def get_left(self):
        left = None
        if self.children:
            left=self.children[0]
        return left
    def get_right(self):
        right = None
        if len(self.children) == 2:
            right=self.children[1]
        return right

    @staticmethod
    def get_height(root):
        if root.children:
            root.height = max(root.get_left().height,root.get_right().height)+1
        else:
            root.height=1
        print root.idx,root.height,'asa'

    @staticmethod
    def get_size(root):
        if root.children:
            root.size = root.get_left().size+root.get_right().size+1
        else:
            root.size=1

    @staticmethod
    def get_spans(root):
        if root.children:
            root.span=root.get_left().span+root.get_right().span
        else:
            root.span=[root.word]

    @staticmethod
    def get_numleaves(self):
        if self.children:
            self.num_leaves=self.get_left().num_leaves+self.get_right().num_leaves
        else:
            self.num_leaves=1

    @staticmethod
    def postOrder(root,func=None,args=None):

        if root is None:
            return
        tNode.postOrder(root.get_left(),func,args)
        tNode.postOrder(root.get_right(),func,args)

        if args is not None:
            func(root,args)
        else:
            func(root)

    @staticmethod
    def encodetokens(root,func):
        if root is None:
            return
        if root.word is None:
            return
        else:
            root.word=func(root.word)

    @staticmethod
    def relabel(root,fine_grained):
        if root is None:
            return
        if root.label is not None:
            if fine_grained:
                root.label += 2
            else:
                if root.label < 0:
                    root.label = 0
                elif root.label == 0:
                    root.label = 1
                else:
                    root.label = 2


def processTree(root,funclist=None,argslist=None):
    if funclist is None:
        root.postOrder(root,root.get_height)
        root.postOrder(root,root.get_num_leaves)
        root.postOrder(root,root.get_size)
    else:
        #print funclist,argslist
        for func,args in zip(funclist,argslist):
            root.postOrder(root,func,args)

    return root

def test_tNode():

    nodes={}
    for i in range(7):
        nodes[i]=tNode(i)
        if i < 4:nodes[i].word=i+10
    nodes[0].parent = nodes[1].parent = nodes[4]
    nodes[2].parent = nodes[3].parent = nodes[5]
    nodes[5].parent = nodes[6].parent = nodes[6]
    nodes[6].add_child(nodes[4])
    nodes[6].add_child(nodes[5])
    nodes[4].add_children([nodes[0],nodes[1]])
    nodes[5].add_children([nodes[2],nodes[3]])
    root=nodes[6]
    postOrder=root.postOrder
    postOrder(root,tNode.get_height,None)
    postOrder(root,tNode.get_numleaves,None)
    postOrder(root,root.get_spans,None)
    print root.height,root.num_leaves
    for n in nodes.itervalues():print n.span

if __name__=='__main__':
    test_tNode()


