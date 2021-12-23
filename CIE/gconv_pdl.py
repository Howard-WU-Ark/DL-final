import math
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Gconv(nn.Layer):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        k = math.sqrt(1.0 / in_features)
        weight_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        weight_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))

        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_1, 
                                bias_attr=bias_attr_1)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_2, 
                                bias_attr=bias_attr_2)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, axis=-2)
    
        '''
        st = time.time()
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        print('Twp Linear layer cost {}s'.format(time.time() - st))
        
        st = time.time()
        x = paddle.bmm(A, F.relu(ax))  # has size (bs, N, num_outputs)
        print('bmm + relu cost {}s'.format(time.time() - st))

        st = time.time()
        x += F.relu(ux)
        print('+= relu() cost {}s'.format(time.time() - st))

        '''
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = paddle.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x

class ChannelIndependentConv(nn.Layer):
    r"""
    Channel Independent Embedding Convolution.
    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    :param in_features: the dimension of input node features
    :param out_features: the dimension of output node features
    :param in_edges: the dimension of input edge features
    :param out_edges: (optional) the dimension of output edge features. It needs to be the same as ``out_features``
    """
    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        super(ChannelIndependentConv, self).__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        # self.node_fc = nn.Linear(in_features, out_features // self.out_edges)
        k = math.sqrt(1.0 / in_features)
        weight_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        self.node_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_1, 
                                bias_attr=bias_attr_1)

        weight_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        self.node_sfc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_2, 
                                bias_attr=bias_attr_2)

        k = math.sqrt(1.0 / in_edges)
        weight_attr_3 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_3 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        self.edge_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_3, 
                                bias_attr=bias_attr_3)

    def forward(self, A, emb_node, emb_edge, mode=1) 
        r"""
        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param emb_node: :math:`(b\times n\times d_n)` input node embedding. :math:`d_n`: node feature dimension
        :param emb_edge: :math:`(b\times n\times n\times d_e)` input edge embedding. :math:`d_e`: edge feature dimension
        :param mode: 1 or 2, refer to the paper for details
        :return: :math:`(b\times n\times d^\prime)` new node embedding,
         :math:`(b\times n\times n\times d^\prime)` new edge embedding
        """
        if mode == 1:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            A = A.unsqueeze(-1)
            A = paddle.mul(A.expand_as(edge_x), edge_x)

            node_x = paddle.matmul(A.transpose((0,3,1,2)),
                                  node_x.unsqueeze(2).transpose((0,3,1,2)))
            node_x = node_x.squeeze(-1).transpose((0,2,1))
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)

            return node_x, edge_x

        elif mode == 2:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            d_x = node_x.unsqueeze(1) - node_x.unsqueeze(2)
            d_x = paddle.sum(d_x ** 2, axis=3, keepdim=False)
            d_x = paddle.exp(-d_x)

            A = A.unsqueeze(-1)
            A = paddle.mul(A.expand_as(edge_x), edge_x)

            node_x = paddle.matmul(A.transpose((0,3,1,2)),
                                  node_x.unsqueeze(2).transpose((0,3,1,2)))
            node_x = node_x.squeeze(-1).transpose((0,2,1))
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)
            return node_x, edge_x

        else:
            raise ValueError('Unknown mode {}. Possible options: 1 or 2'.format(mode))

class Siamese_Gconv(nn.Layer):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2

class Siamese_ChannelIndependentConv(nn.Layer):
    r"""
    Siamese Channel Independent Conv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    :param in_edges: the dimension of input edge features
    :param out_edges: (optional) the dimension of output edge features. It needs to be the same as ``num_features``
    """
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super(Siamese_ChannelIndependentConv, self).__init__()
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def forward(self, g1: Tuple[Tensor, Tensor, Optional[bool]], *args) -> List[Tensor]:
        r"""
        Forward computation of Siamese Channel Independent Conv.

        :param g1: The first graph, which is a tuple of (:math:`(b\times n\times n)` {0,1} adjacency matrix,
         :math:`(b\times n\times d_n)` input node embedding, :math:`(b\times n\times n\times d_e)` input edge embedding,
         mode (``1`` or ``2``))
        :param args: Other graphs
        :return: A list of tensors composed of new node embeddings :math:`(b\times n\times d^\prime)`, appended with new
         edge embeddings :math:`(b\times n\times n\times d^\prime)`
        """
        emb1, emb_edge1 = self.gconv(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv(*g)
            embs.append(emb2), emb_edges.append(emb_edge2)
        return embs + emb_edges