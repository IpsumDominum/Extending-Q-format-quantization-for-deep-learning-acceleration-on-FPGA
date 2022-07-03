# Extending-Q-format-quantization-for-deep-learning-acceleration-on-FPGA
----

### Abstract

Deep Learning has demonstrated great success in many fields of AI such as
computer vision, natural language processing and more. Though unfortunately high
performing models are costly to compute and consumes a lot of energy. Especially
in recent years, it is observed that extremely large deep learning models may have a
qualitative advantage to their smaller counter parts. It is therefore a very important
question to ask how neuralnetwork computations can be accelerated all the while con-
suming less energy. Graphic processing Units, or GPUs, have been traditionally used
to accelerate neuronetwork computation due to its SIMD capabilities. However, GPUs
were originally designed to process computer graphic applications and thus can carry
much unneeded overhead when computing deep neuronetworks. Application-specific
integrated circuit(ASIC) consume much less power than its GPU counterparts but are
not easy to design, build and test rapidly. Field programmable gate arrays (FPGAs) are
a more flexible alternative to ASIC allowing for rapid design and testing. FPGAs are
slightly slower than ASICs but retains its efficiency advantage over GPUs, thus making
it a popular platform to conduct neuronetwork hardware acceleration experiments.
In particular, FPGAs is often used to co-design software and hardware methods so
they work better combined leveraging each other. Quantization is a simple, scalable,
and accuracy preserving method usually applied in software but may be combined
efficiently with hardware optimizations. Work by Chun Yan Lo and Chiu-Wing Sham
showed that a well optimized fixed point quantization scheme is capable of 20x speed
up in Giga Operations Per second compared with Nvidia RTX 2070 GPU running
neuronetwork with the keras library.
One difference however between the Lo et al implementation and traditional quanti-
zation is that they use a special quantization format called Q-format. Instead of using
full integer quantization Q-format delegates parts of the bits to after the binary point.
Whilst traditional quantization only optimize matrix multiplication and still requires
some floating point calculation as a result of scaling, the Q-format quantization uses
one single data type without needing conversions back and forth, making the hardware
implementation straightforward and with reduced overhead.
The aim of this thesis is to extend the Q-format quantization scheme to more com-
plex model types including an augmented version of the LeNet-5 architecture used in Lo
et al, with added Batch normalization layers, as well as a transformer network trained
on a subset of the Europarl dataset. The effect of quantization aware training as well as
post quantization finetuning combined with Q-format quantization is also tested.


