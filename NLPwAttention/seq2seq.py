import trax
import numpy as np
from trax.fastmath import numpy


## encoder

##atention

##decoder
    #pre attention decoder - provides hidden states to att mechanism
    #post-attention decoder - provides the translation

### NOTE: The decoder is supposed to pass his previous hidden states to the att. mechanism to get context vectors


## Hidden states from the Encoder are used as k and V
## hidden states from the Pre-att Decoder are used as Q
## the att mechanism outputs Context vectors

# [Encoder(input_seq), pre-att-Decoder(target_seq)] -> [Q,K,V] >> att-mechanism  -> context-vectors
# [context-vectors]  >> post-att-decoder -> predicted-Seq 
