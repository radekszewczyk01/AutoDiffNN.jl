module AutoDiff

using LinearAlgebra

# Ładujemy podmoduły
include("models.jl")
include("operators.jl")
include("functions.jl")

# Eksportujemy rzeczy, które mają być dostępne na zewnątrz
export GraphNode, Operator, Constant, Variable, ScalarOperator,
       BroadcastedOperator, topological_sort, forward!, backward!, update!

end
