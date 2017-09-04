from LDPCEncoder import LDPCEncoder
from LDPCDecoder import LDPCDecoder

n = 4096
message_length, code_length, codeword_mat, ldpc_code, encode_mat = LDPCEncoder(n, 0.2)
# message_to_send is an integer from 0 to n-1.
message_to_send = 1234
# Select the message_to_send-th column.
codeword = codeword_mat[:, message_to_send]
received_codeword = codeword.copy()
# Flip some bits in the received_codeword
# flip probability = 0.05
# We should flip each bit in the codeword with probability 0.05, independently.
# However, I am being lazy here.
for i in xrange(4, code_length, 20):
	received_codeword[i] = 1 - codeword[i]

decode_result = LDPCDecoder(n, ldpc_code, encode_mat, received_codeword, 0.05)

if decode_result == message_to_send:
	print 'Decoding correct'
else:
	print 'Decoding error'
