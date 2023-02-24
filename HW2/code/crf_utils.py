# import time
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reverse_onehot(labels) :
	mask = torch.arange(26, dtype=torch.float).to(device)		## Number encoding of each letter
	masked_labels = torch.zeros(len(labels), dtype=torch.float).to(device)
	for i in range(len(labels)):
		# print("SHAPEEEEEES: ", mask.shape, labels[i].shape)
		# print(torch.mm(mask,labels[i]))
		masked_labels[i] = torch.mm(mask.reshape(1,-1),labels[i].reshape(-1,1)).item()
	return masked_labels

def onehot(masked_labels, num_labels) :
	label_dict = torch.eye(num_labels, dtype=torch.float).to(device)						## 26x26 identity matrix. Each ith row is one-hot representation of ith letter
	# print(label_dict.shape)
	# number of letters of a word
	m = len(masked_labels)
	labels = torch.zeros((m, num_labels)).to(device)
	# print(labels.shape)
	for i in range(m):
		labels[i] = label_dict[masked_labels[i]]
	# print(labels.shape)
	return labels

def computeAllDotProduct(w, data):
	dots = torch.mm(w, torch.transpose(data, 0, 1)).to(device)
	return dots

def logTrick(numbers):

	if len(numbers.shape) == 1:
		M = torch.max(numbers).to(device)
		return M + torch.log(torch.sum(torch.exp(numbers - M)))
	else:
		M = torch.max(numbers, 1).values
		temp = torch.transpose(numbers, 0, 1) - M
		return M + torch.log(torch.sum(torch.exp(torch.transpose(temp, 0, 1)), 1))

def get_index(t, i):
	return(int(t[i].item()))


def logPYX(label, w, T, alpha, dots):
	masked_label = reverse_onehot(label)
	m = len(masked_label)
	res = sum([dots[get_index(masked_label, i), i] for i in range(m)]) + sum([T[get_index(masked_label,i), get_index(masked_label,i + 1)] for i in range(m - 1)])
	logZ = logTrick(dots[:, m - 1] + alpha[m - 1, :])
	res -= logZ

	return res

def computeDP(m, w, T, dots, num_labels):

	alpha = torch.zeros((m, num_labels), dtype=torch.float).to(device)
	for i in range(1, m):
		alpha_one = dots[:, i - 1] + alpha[i - 1, :]
		alpha_one = alpha_one.repeat(num_labels, 1) + torch.transpose(T, 0, 1)
		alpha[i] = logTrick(alpha_one)
	beta = torch.zeros((m, num_labels)).to(device)
	for i in range(m - 2, -1, -1):
		beta_one = dots[:, i + 1] + beta[i + 1, :]
		beta_one = beta_one.repeat(num_labels, 1) + T
		beta[i] = logTrick(beta_one)

	return alpha, beta

def obj_func(features, labels, params, C, num_labels, embed_dim) :
	w = (params[ : embed_dim * num_labels]).reshape(num_labels, embed_dim).clone().to(device)
	T = (params[embed_dim * num_labels : ]).reshape(num_labels, num_labels).clone().to(device)
	meanLogPYX = 0
	for data,label in zip(features,labels) :
		data, label = data.to(device), label.to(device)
		m = len(label)
		dots = computeAllDotProduct(w, data)
		alpha, beta = computeDP(m, w, T, dots, num_labels)
		meanLogPYX += logPYX(label, w, T, alpha, dots)
	meanLogPYX /= len(features)

	objValue = -C * meanLogPYX + 0.5 * torch.sum(w ** 2) + 0.5 * torch.sum(T ** 2)
	print(objValue)
	return objValue

def dp_infer(features, params, num_labels, embed_dim):
	w = (params[ : embed_dim * num_labels]).reshape(num_labels, embed_dim).clone().to(device)
	T = (params[embed_dim * num_labels : ]).reshape(num_labels, num_labels).clone().to(device)
	
	batch_size = len(features)
	results = torch.empty(batch_size, len(features[0]), num_labels).to(device)
	for i_word, x in enumerate(features) :
		x = x.to(device)
		m = len(x)					## number of letters in word x
		pos_letter_value_table = torch.zeros((m, num_labels), dtype=torch.float64).to(device)
		pos_best_prevletter_table = torch.zeros((m, num_labels), dtype=torch.int).to(device)

		# for the position 1 (1st letter), special handling
		# because only w and x dot product is covered and transition is not considered.
		for i in range(num_labels):
			# print(w)
			# print(x)
			# print(w[i, :].reshape(1,-1).shape,x[0, :].reshape(-1,1).shape)
			pos_letter_value_table[0, i] = torch.mm(w[i, :].reshape(1,-1), x[0, :].reshape(-1,1))

		# pos_best_prevletter_table first row is all zero as there is no previous letter for the first letter

		# start from 2nd position
		for pos in range(1, m):
			# go over all possible letters
			for letter_ind in range(num_labels):
				# get the previous letter scores
				prev_letter_scores = (pos_letter_value_table[pos-1, :]).clone().to(device)
				# we need to calculate scores of combining the current letter and all previous letters
				# no need to calculate the dot product because dot product only covers current letter and position
				# which means it is independent of all previous letters
				for prev_letter_ind in range(num_labels):
					prev_letter_scores[prev_letter_ind] += T[prev_letter_ind, letter_ind]

				# find out which previous letter achieved the largest score by now
				best_letter_ind = torch.argmax(prev_letter_scores).to(device)
				# update the score of current positive with current letter
				pos_letter_value_table[pos, letter_ind] = prev_letter_scores[best_letter_ind] + torch.mm(w[letter_ind,:].reshape(1,-1), x[pos, :].reshape(-1,1))
				# save the best previous letter for following tracking to generate most possible word
				pos_best_prevletter_table[pos, letter_ind] = best_letter_ind
		letter_indicies = torch.zeros((m, 1), dtype=torch.long).to(device)
		letter_indicies[m-1, 0] = torch.argmax(pos_letter_value_table[m-1, :])
		max_obj_val = pos_letter_value_table[m-1, letter_indicies[m-1, 0]]
		# print(max_obj_val)
		for pos in range(m-2, -1, -1):
			letter_indicies[pos, 0] = pos_best_prevletter_table[pos+1, letter_indicies[pos+1, 0]]
		## TODO : collect letter indices for the batch and return the batch
		# One-hot encode targets.
		word_predict = onehot(letter_indicies, num_labels)
		results[i_word] = word_predict
	
	return results
		# target = np.zeros(dataset.target.shape + (26,))
		# for index, letter in np.ndenumerate(results):
		# 	if letter:
		# 		target[index][ord(letter) - ord('a')] = 1
		# results = target

if __name__ == "__main__":
	# print(onehot([0, 7], 26))
	labels = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).to(device)
	print(labels.shape)
	print(reverse_onehot(labels))
	# a = torch.tensor([1, 0, 3])
	# b = torch.tensor([1, 2, 4])
	# print(a.shape, b.shape)
	# c = torch.mm(a.reshape(1,-1), b.reshape(-1,1))
	# print(c.item())



def computeMarginal(m, w, T, alpha, beta, dots, num_labels):

	p1 = torch.zeros((m, num_labels)).to(device)
	for i in range(m):
		p1[i] = alpha[i, :] + beta[i, :] + dots[:, i]
		p1[i] = torch.exp(p1[i] - logTrick(p1[i]))
	p2 = torch.zeros((m - 1, num_labels, num_labels)).to(device)
	for i in range(m - 1):
		a = alpha[i, :] + dots[:, i]
		a = a.repeat((num_labels, 1))
		a = torch.transpose(a, 0, 1)

		b = beta[i + 1, :] + dots[:, i + 1]
		b = b.repeat(num_labels, 1)
		p2[i] = a + b + T	### TODO
		p2[i] = torch.exp(p2[i] - logTrick(p2[i].flatten()))

	return p1, p2

def computeGradientWy(data, label, p1, num_labels):

	m = len(label)
	cof = torch.zeros((num_labels, m)).to(device)
	for i in range(m):
		cof[label[i], i] = 1
	cof -= torch.transpose(p1)
	res = torch.mm(cof, data)

	return res

def computeGradientTij(label, p2):

	m = len(label)
	res = torch.zeros(p2.shape).to(device)
	for i in range(m - 1):
		res[i, label[i], label[i + 1]] = 1
	res -= p2
	res = torch.sum(res, 0)
   
	return res

def crfFuncGrad(features, labels, params, C, num_labels, embed_dim):

	w = (params[ : embed_dim * num_labels]).reshape(num_labels, embed_dim).clone().to(device)
	T = (params[embed_dim * num_labels : ]).reshape(num_labels, num_labels).clone().to(device)
	
	meandw = torch.zeros((num_labels, embed_dim)).to(device)
	meandT = torch.zeros((num_labels, num_labels)).to(device)

	for word, label in zip(features, labels):

		m = len(word)
		dots = computeAllDotProduct(w, word)
		alpha, beta = computeDP(m, w, T, dots, num_labels)
		p1, p2 = computeMarginal(m, w, T, alpha, beta, dots, num_labels)

		dw = computeGradientWy(word, label, p1, num_labels)
		dT = computeGradientTij(label, p2)

		meandw += dw
		meandT += dT

	meandw /= len(features)
	meandT /= len(features)

	meandw *= (-C)
	meandT *= (-C)

	meandw += w
	meandT += T

	gradients = torch.cat((meandw.flatten(), meandT.flatten()))

	return gradients
