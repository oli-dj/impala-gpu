# impala-gpu
MPS algorith using the principles of SNESIM / IMPALA, implemented in MATLAB with core parts accellerated by GPU.
GPU implementation can handle conditioning to hard data, soft data (both colocational and not).
CPU implementation does not handle non-colocational soft data yet.

Developed in MATLAB R2017b, not tested in earlier versions.
GPU code requires CUDA installed and is tested to work with NVIDIA Pascal or compatible graphicscards.
Earlier cards may work. Most cards will work, but may require compiling of the .cu files for their architiceture.

Requires https://github.com/cultpenguin/mGstat  in path for mps_template.m and channels.m.

## Installation
Clone and use directly from MATLAB, see example_script.m for an example.

Set "options.GPU = 0;" if no CUDA capable GPU avaliable, note: will not handle non-colocational soft data in this case.

