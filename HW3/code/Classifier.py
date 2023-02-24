
import torch
import torch.nn as nn
import torch.autograd as ag
import ProxLSTM as pro

from torch.autograd import Variable
from torch.nn import functional as F



class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, input_size):
		super(LSTMClassifier, self).__init__()
		

		self.output_size = output_size	# should be 9
		self.hidden_size = hidden_size  #the dimension of the LSTM output layer
		self.input_size = input_size	  # should be 12
		# self.normalize = F.normalize()
		self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= 64, kernel_size= 10, stride= 3) # feel free to change out_channels, kernel_size, stride
		self.relu = nn.ReLU()
		self.lstm = nn.LSTM(64, hidden_size, batch_first = True)
		self.lstmcell = nn.LSTMCell(input_size= 64, hidden_size= hidden_size)
		# self.ProxLSTMCell = pro.ProximalLSTMCell(self.lstmcell, input_size= 64, hidden_size= hidden_size)
		# self.ProxLSTMCell = pro.ProximalLSTMCell(self.lstmcell)
		self.linear = nn.Linear(self.hidden_size, self.output_size)
		self.apply_dropout = False
		self.apply_batch_norm = False
		self.dropout = nn.Dropout()
		self.batch_norm = nn.BatchNorm1d(64)

		self.h_t = torch.zeros(self.hidden_size, requires_grad= True)		# h_0
		self.c_t = torch.zeros(self.hidden_size, requires_grad= True)		# c_0
			

		
	def forward(self, input, r, batch_size, mode='plain', prox_epsilon=1, epsilon=0.01):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size
		
		if mode == 'plain' :
			normalized = F.normalize(input)
			embedding = self.conv(normalized.permute(0,2,1)).permute(2,0,1)
			self.lstm_input = self.relu(embedding)
			self.h_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# h_0
			self.c_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# c_0
			for seq in self.lstm_input:
				self.h_t, self.c_t = self.lstmcell(seq, (self.h_t, self.c_t))
			decoded = self.linear(self.h_t)
				
		
		if mode == 'AdvLSTM' :
			normalized = F.normalize(input)
			embedding = self.conv(normalized.permute(0,2,1)).permute(2,0,1)
			self.lstm_input = self.relu(embedding) + (epsilon * r)
			self.h_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# h_0
			self.c_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# c_0
			for seq in self.lstm_input :
				self.h_t, self.c_t = self.lstmcell(seq, (self.h_t, self.c_t))
			decoded = self.linear(self.h_t)

		
		if mode == 'ProxLSTM' :
			prox = pro.ProximalLSTMCell.apply
			normalized = F.normalize(input)
			# Dropout layer
			if self.apply_dropout:
				normalized = self.dropout(normalized)
			embedding = self.conv(normalized.permute(0,2,1)).permute(2,0,1)
			self.lstm_input = self.relu(embedding)
			# Batch Norm layer
			if self.apply_batch_norm:
				self.lstm_input = self.batch_norm(self.lstm_input.permute(0, 2, 1))
				self.lstm_input = self.lstm_input.permute(0, 2, 1)
			self.h_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# h_0
			self.c_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# c_0
			for seq in self.lstm_input:
				self.h_t, self.s_t = self.lstmcell(seq,(self.h_t, self.c_t))
				self.G_t = torch.zeros(seq.shape[0], self.lstmcell.hidden_size, self.lstmcell.input_size)
				for i in range(self.s_t.size(-1)):
					g_t = ag.grad(self.s_t[:,i], seq, grad_outputs=torch.ones_like(self.s_t[:,0]), retain_graph=True)[0]
					# print("g_t: ", g_t)
					self.G_t[:,i,:] = g_t[0]
					
				# print("G_t.shape", self.G_t.shape)
				self.h_t, self.c_t = prox(self.h_t, self.s_t, self.G_t, prox_epsilon)
			decoded = self.linear(self.h_t)
		
		return decoded


	def get_lstm_input(self):
		return self.lstm_input
		
# X = torch.tensor([[[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12]],
# 			  [[21,22,23,24],
# 			  [25,26,27,28],
# 			  [29,30,31,32]]], dtype=torch.float32)
# print(X)
# Y = torch.flatten(X,0,1)
# print(Y)
# Z = Y.reshape(2,3,4)
# print(Z)
# ZZ = Z[:,-1]
# print(ZZ,ZZ.shape)
# model = nn.Linear(4, 9)
# p = model(Z[:,-1])
# print(p,p.shape)
# q = torch.argmax(p, dim = 1)
# print (q.shape)

# A = torch.tensor([2, 2, 8, 2, 4, 4, 4, 4, 4, 7, 7, 7, 8, 1, 2, 2, 4, 4, 4, 4, 7, 7, 7, 7,
#         7, 7, 7])
# print(A.view(-1,1))
# input = torch.randn(6, 3, 10)
# for seq in input :
# 	print (seq.shape[0],50)