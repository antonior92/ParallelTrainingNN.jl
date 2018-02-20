# ParallelTrainingNN

[![Build Status](https://travis-ci.org/antonior92/ParallelTrainingNN.jl.svg?branch=master)](https://travis-ci.org/antonior92/ParallelTrainingNN.jl)
[![Coverage Status](https://coveralls.io/repos/antonior92/ParallelTrainingNN.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/antonior92/ParallelTrainingNN.jl?branch=master)
[![codecov.io](http://codecov.io/github/antonior92/ParallelTrainingNN.jl/coverage.svg?branch=master)](http://codecov.io/github/antonior92/ParallelTrainingNN.jl?branch=master)

This package allows to model dynamic systems using
feedforward neural networks. The neural networks
may be trained using parallel or series-parallel
training.


## Installation

Within [Julia](https://julialang.org/downloads/), use the package manager:

```JULIA
julia> Pkg.clone("https://github.com/antonior92/ParallelTrainingNN.jl")
```

The package installation can be tested using the command:

```JULIA
julia> Pkg.test("ParallelTrainingNN")
```

## Reference

Both the implementation and the examples are originally from the paper:
```
""Parallel Training Considered Harmful?": Comparing Series-Parallel and Parallel Feedforward Network Training"
Antonio H. Ribeiro and Luis A. Aguirre
```
Preprint available in arXiv ([here](https://arxiv.org/abs/1706.07119.pdf))

BibTeX entry:
```
@article{DBLP:journals/corr/RibeiroA17,
  author    = {Ant{\^{o}}nio H. Ribeiro and
               Luis A. Aguirre},
  title     = {"Parallel Training Considered Harmful?": Comparing Series-Parallel
               and Parallel Feedforward Network Training},
  journal   = {CoRR},
  volume    = {abs/1706.07119},
  year      = {2017},
  url       = {http://arxiv.org/abs/1706.07119},
  archivePrefix = {arXiv},
  eprint    = {1706.07119},
  timestamp = {Mon, 03 Jul 2017 13:29:02 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/RibeiroA17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

## Examples

Folder [``examples``](https://github.com/antonior92/ParallelTrainingNN.jl/tree/master/examples) contain Jupyter notebooks for reproducing the examples presented in the paper.
