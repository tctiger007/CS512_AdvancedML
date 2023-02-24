import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    # def __init__(self, lstm):	# feel free to add more input arguments as needed
        # super(ProximalLSTMCell, self).__init__()
        # lstm_cell = lstm   # use LSTMCell as blackbox

    @staticmethod
    def forward(ctx, h_t, s_t, G_t, prox_epsilon=1):
        '''need to be implemented'''
        # print("input.shape: ", input.shape)
        # with torch.enable_grad():
            # G_t = torch.zeros(input.shape[0], lstm_cell.hidden_size, lstm_cell.input_size)
        #     h_t, s_t = lstm_cell(input,(pre_h, pre_c))
            
        # print(h_t.backward(torch.ones_like(h_t), retain_graph = True))
        # grad_ht_ht1 = pre_h.grad
        # grad_ht_ct1 = pre_c.grad
        # s_t.backward(torch.ones_like(s_t), retain_graph = True)
        # grad_st_ht1 = pre_h.grad
        # grad_st_ct1 = pre_c.grad
        
        # print(grad_ht_ht1)

        # G_t = torch.zeros(input.shape[0], lstm_cell.hidden_size, lstm_cell.input_size)
        # for i in range(s_t.size(-1)):
        #     g_t = ag.grad(s_t[:,i], input, grad_outputs=torch.ones_like(s_t[:,0]), retain_graph=True)
        #     G_t[:,i,:] = g_t[0]
        s_t = s_t.unsqueeze(2)
        G_t_transpose = torch.transpose(G_t, 1, 2)            #G_t.permute(1,0)
        mul = torch.matmul(G_t, G_t_transpose)
        my_eye = torch.eye(mul.shape[-1])
        my_eye = my_eye.reshape((1, my_eye.shape[0], my_eye.shape[0]))
        my_eye = my_eye.repeat(h_t.shape[0], 1, 1)
        inverse = torch.inverse(my_eye + prox_epsilon*mul)
        c_t = torch.matmul(inverse, s_t)
        c_t = c_t.squeeze()
        ctx.save_for_backward(h_t, c_t, G_t, inverse)

        return (h_t, c_t)


    @staticmethod
    def backward(ctx, grad_h, grad_c):
        '''need to be implemented'''
        # grad_input = grad_pre_c = grad_pre_h = None
        h_t, c_t, G_t, inverse = ctx.saved_tensors
        # print("inverse.shape: {}, grad_c.shape: {}".format(inverse.shape, grad_c.unsqueeze(2).shape))
                
        
        
        a = torch.matmul(inverse,grad_c.unsqueeze(2))
        # a = a.squeeze()
        # a_transpose = torch.transpose(a, 1, 2)
        # print("a.shape: ", a.shape)
        # print("c_t.shape: ", c_t.unsqueeze(2).shape)
        # print("transpose shape: ", c_t.unsqueeze(2).permute(0, 2, 1).shape)
        # print(torch.matmul(a, c_t.unsqueeze(2).permute(0, 2, 1)))
        grad_g1 = torch.matmul(a, c_t.unsqueeze(2).permute(0, 2, 1))
        grad_g2 = torch.matmul(c_t.unsqueeze(2), a.permute(0, 2, 1))
        # print("========= : ", (grad_g1 + grad_g2).shape)
        # print("G_t.shape: ", G_t.shape)
        grad_g = -torch.matmul(grad_g1 + grad_g2, G_t)
        # grad_g = -torch.matmul((torch.matmul(a, c_t.unsqueeze(2).permute(0, 2, 1)) + torch.matmul(c_t, a.transpose())), G_t)
        
        # print("grad_c.shape: ", grad_c.unsqueeze(2).shape)
        # print("inverse.shape", inverse.shape)
        grad_s = torch.matmul(grad_c.unsqueeze(2).permute(0, 2, 1), inverse)
        grad_s = grad_s.squeeze()

        # print("grad_g.shape:" , grad_g.shape)
        # print("grad_h.shape:" , grad_h.shape)
        # print("grad_s.shape:" , grad_s.shape)
        # print("grad_c: ", grad_c)
        # grad_g = torch.zeros(27,64)

        return grad_h, grad_s, grad_g, None
        

