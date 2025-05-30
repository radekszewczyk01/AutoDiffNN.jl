module AutoDiff

import Base: show, summary
import NNlib
using LinearAlgebra
using NNlib: conv, DenseConvDims,∇conv_data, ∇conv_filter
# __precompile__(false)

include("models.jl")
include("operators.jl")
include("functions.jl")
include("cnn_operators.jl")

export GraphNode, Operator, Constant, Variable, ScalarOperator,
       BroadcastedOperator, topological_sort, forward!, backward!, update!, relu, σ, swish, linear,
       softmax, PermuteDims, ConvOperator, MaxPoolOperator, EmbeddingOperator, FlattenOperator
end
