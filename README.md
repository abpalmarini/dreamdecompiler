# DreamDecompiler
A record of the code and experiments for the paper:
> Palmarini, Alessandro B., Christopher G. Lucas, and N. Siddharth. "Bayesian Program Learning by Decompiling Amortized Knowledge." Forty-first International Conference on Machine Learning. 2024. [[arXiv]](https://arxiv.org/abs/2306.07856)

This repository contains a fork of the `dreamcoder` subfolder in the original [DreamCoder repository](https://github.com/ellisk42/ec). The changes, mainly located in `dreamcoder/dreamdecompiler.py`, integrate the alternative variants for dream decompiling (as presented in the paper) within the [DreamCoder](https://arxiv.org/abs/2006.08381) system.

## Usage
To be used within the original [DreamCoder repository](https://github.com/ellisk42/ec).

To run on a new machine, you will need to do the following.

Create a shallow clone of the original repository:
```
git clone --depth 1 -b master https://github.com/ellisk42/ec.git
```

If you don't have ssh keys set up you will need to first change their `.gitmodules` file to contain the following:
```
[submodule "pregex"]
	path = pregex
	url = https://github.com/insperatum/pregex
	branch = master
[submodule "pinn"]
	path = pinn
	url = https://github.com/insperatum/pinn
	branch = master
[submodule "pyccg"]
	path = pyccg
	url = https://github.com/hans/pyccg
	branch = master
```

Clone the submodules:
```
git submodule update --recursive --init
```

Remove the `dreamcoder` subdirectory and replace it with the modified version found in this repository:
```
rm -rf ec/dreamcoder
git clone https://github.com/abpalmarini/dreamdecompiler.git
mv dreamdecompiler/dreamcoder ec/
```

Follow the instructions provided in the [original repository's README](https://github.com/ellisk42/ec) to
setup the necessary dependencies.

To chunk with the DreamDecompiler-PC model, add flags `--compressor ddc_vs  --chunkWeighting raw --numConsolidate`  to any DreamCoder command. Remove `--chunkWeighting raw` to use DreamDecompiler-Avg. 

The commands for the experiments mentioned in the paper can be found in `experiments.sh`.
