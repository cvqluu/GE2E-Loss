# GE2E-Loss
Pytorch implementation of Generalized End-to-End Loss for speaker verification, proposed in https://arxiv.org/pdf/1710.10467.pdf [1].

Includes an argument to define whether to use the 'softmax' or 'contrast' type loss (equations 6 and 7 respectively in [1]). Uses vector operations to speed up calculations of the cosine similarity scores for an utterance embedding against all the other speaker embedding centroids.

Below is some example code for how to use this. The example values for certain parameters are taken from [1]

```python

import torch
from ge2e import GE2ELoss

criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax') #for softmax loss
criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='contrast') #for contrast loss

N = 64 #Number of speakers in a batch
M = 10 #Number of utterances for each speaker
D = 256 #Dimensions of the speaker embeddings, such as a d-vector or x-vector

test_input = torch.rand(N, M, D)
loss = criterion(test_input) #output is a scalar
loss.backward()
```

[1] GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION, https://arxiv.org/pdf/1710.10467.pdf
