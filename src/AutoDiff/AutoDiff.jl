module AutoDiff

import Base: show, summary
using LinearAlgebra
using NNlib
# __precompile__(false)

include("models.jl")
include("operators.jl")
include("functions.jl")


export GraphNode, Operator, Constant, Variable, ScalarOperator,
       BroadcastedOperator, topological_sort, forward!, backward!, update!, relu, σ, swish, linear,
       softmax, EmbeddingOp, PermuteDimsOp, ReshapeOp, FlattenOp, Conv1DOp, MaxPool1DOp
end
