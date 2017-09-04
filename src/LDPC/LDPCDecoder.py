from pycodes.pyLDPC import LDPCCode
from pycodes.utils.CodeMaker import make_H_gallager
from pycodes.utils.channels import BSCLLR
import pyldpc
from numpy import *
import numpy as np

'''
LDPC Decoder
inputs:
	n: number of messages
	ldpc_code: an ldpc code object
	G: ldpc encode matrix
	received_code: the received codeword, the codeword may be flipped in a BSC channel
	flip_p: the bit flip probability of BSC; if don't know exact bit flip probability,
			give an estimate
returns:
	Result of decoding, i.e., the index of original message.
'''

def LDPCDecoder(n, ldpc_code, G, received_code, flip_p = 0.3):
	evidence = BSCLLR(received_code, flip_p)
	ldpc_code.setevidence(evidence, alg='SumProductBP')
	for iteration in range(100):
		ldpc_code.decode()
	beliefs = ldpc_code.getbeliefs()
	transmitted_code = map(lambda x: x > 0.5, beliefs)
	message = pyldpc.DecodedMessage(G, transmitted_code)
	log_n = int(math.ceil(math.log(n, 2)))
	pow2 = 1
	result = 0
	for i in xrange(log_n):
		result += message[i] * pow2
		pow2 *= 2
	return result

