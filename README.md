# Deltanet
this is an implementation of the paper "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (https://arxiv.org/abs/2406.06484) 
it implement also the paper "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues" (https://arxiv.org/abs/2411.12537)

# Requirement 
All you need is pytorch

# How To use
Just import the deltanet and use the DeltaBlock in your code.
There is an example just look inside.

# Why does this exist when there is already an Official implementation?
The official implementation is faster and more optimized, however the code is complex. I am using this to hack the model and try many experimentation (ex: using a Bitnet Matrix instead of float Matrix). 

# Disclaimer
I have not tested this on cuda but it should work.
