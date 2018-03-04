from numpy import *
import numpy as np
import pyldpc
from pycodes.pyLDPC import LDPCCode

'''
LDPC Encoder
inputs:
	n: number of messages to encode. We use integers 0:n-1 to represent the n messages
	rate: the rate of LDPC code, float number in (0, 1)
returns:
	message_length: length (in bits) of the encoded messages
	code_length: length (in bits) of the codewords
	codeword_mat: a matrix of size (code_length, n). The i-th column stores the 
				  codeword for the i-th message
	ldpc_code: an ldpc code object
	G: ldpc encode matrix
'''

def LDPCEncoder(n, rate = 0.5):
	dv = 3
	dc = 6
	log_n = int(math.ceil(math.log(n, 2)))
	flag = False
	H = 0
	G = 0
	code_length = 0

	while flag == False:
		code_length = int(math.ceil(log_n/float(rate)))
		if code_length % dc != 0:
			# dc must divide code_length
			code_length += (dc - code_length % dc)
		H = pyldpc.RegularH(code_length, dv, dc)
		G = pyldpc.CodingMatrix(H)
		if G.shape[1] >= log_n:
			flag = True
		else:
			rate = rate/2.0

	# Now H is the binary check matrix and G is the encoding matrix
	# Size of G is (code_length, message_length)
	message_length = G.shape[1]
	# Convert integers in xrange(n) to binary numbers
	binary_mat = zeros([message_length, n], dtype = int)
	for i in xrange(n):
		bin_str = np.binary_repr(i, log_n)
		for j in xrange(log_n):
			binary_mat[j, i] = ord(bin_str[log_n - j - 1])-48

	codeword_mat = (dot(G, binary_mat)) % 2

	check_node_num = code_length * dv / dc
	L = []
	for i in range(check_node_num):
		non_zero_pos = np.where(H[i, :] != 0)
		L.append((non_zero_pos[0]).tolist())

	E = reduce(lambda x,y:x+y, map(lambda z:len(z), L))
	ldpc_code = LDPCCode(code_length, check_node_num, E, L)

	return message_length, code_length, codeword_mat, ldpc_code, G

