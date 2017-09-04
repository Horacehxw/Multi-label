from pycodes.pyLDPC import LDPCCode
from pycodes.utils.CodeMaker import make_H_gallager
from pycodes.utils.channels import BSCLLR
import pyldpc
from numpy import *
import numpy as np


class LDPCCoder():
    '''
    athor: Horace He. 2017.9.4
    LDPC Coder
    Generate LDPC decode and encode class.
    Attribute:
        message_length: length (in bits) of the encoded messages
    	code_length: length (in bits) of the codewords
    	codeword_mat: a matrix of size (code_length, n). The i-th column stores the
    				  codeword for the i-th message
    	ldpc_code: an ldpc code object
    	encode_mat: ldpc encode matrix

    Function:
        encode: take in a bit string and encode them iteratively through bits-length block
        decode: decode the encoded bit string by block length

    Usage:
    >> ldpc = LDPCCoder(10)
    >> received_codeword = ldpc.encode([0,0,0,1,0,1,1,1,1,1])
    >> for i in xrange(len(received_codeword)):
    ....   if np.random.rand() > 0.95:
    ....   received_codeword[i] = 1 - received_codeword[i]
    >> ldpc.decode(received_codeword)
    array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    '''
    def __init__(self, bits, rate=0.5,flip_p=0.3):
        '''
        create a LDPCCoder
        '''
        n = 1<<bits
        self.bits = bits
        self.flip_p = flip_p
        self.message_length, self.code_length, self.codeword_mat, self.ldpc_code, self.encode_mat = __LDPCGenerate__(n, rate)
        #self.encode, self.decode = __LDPCCoder__(self.bits, self.ldpc_code, self.encode_mat, self.codeword_mat, self.flip_p)

    def encode(self, message):
        '''
        encode the binary vector for every bits-length block
        input:
            message: binary vector
        return:
            encoded: encoded binary vector
        '''
        def bin_to_num(x):
            pow2=1
            num = 0
            for binary in x:
                num += pow2*binary
                pow2*=2
            return num

        encoded = None
        bits = self.bits
        for i in range(0, len(message), bits):
            message_encode = self.__encode(bin_to_num(message[i:i+bits]))
            if i == 0:
                encoded = message_encode
            else:
                encoded = np.concatenate((encoded, message_encode), axis=0)
        return encoded

    def decode(self, message):
        '''
        decode the binary received message for every code_length-length block
        input:
            message: encoded message
        return:
            encoded: decoded  binary vector
        Wraning:
            if (message length % bits != 0) at the initialization phase:
                the return decoded vector may have some recundent bits at the end compared to the origin message
                It's safe to just delete the redundent tails
        '''
        decoded = None
        step = self.code_length
        for i in range(0, len(message), step):
            message_decode = self.__decode(message[i:i+step])
            if i == 0:
                decoded = message_decode
            else:
                decoded = np.concatenate((decoded, message_decode), axis=0)
        return decoded


    def __decode(self, received_code):
        bits, ldpc_code, encode_mat, flip_p = \
            self.bits, self.ldpc_code, self.encode_mat, self.flip_p
        evidence = BSCLLR(received_code, flip_p)
        ldpc_code.setevidence(evidence, alg='SumProductBP')
        for iteration in range(100):
            ldpc_code.decode()
        beliefs = ldpc_code.getbeliefs()
        transmitted_code = map(lambda x: x > 0.5, beliefs)
        message = pyldpc.DecodedMessage(encode_mat, transmitted_code)
        return message[0:bits]

    def __encode(self, origin_message):
        codeword_mat = self.codeword_mat
        return codeword_mat[:,origin_message].copy()

def __LDPCGenerate__(n, rate = 0.5):
    '''
    LDPC Generate
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
