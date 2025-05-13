using Test
using Random
using myExample  # Załaduj bibliotekę zawierającą AutoDiff i MiniFlux

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux
@testset "Embedding layer forward and backward pass" begin
    vocab_size = 10
    embedding_dim = 4
    x_val = [1, 2, 3]
    x = AD.Variable(x_val, name="x")

    embedding_layer = MF.Embedding(vocab_size, embedding_dim)
    y = embedding_layer(x)
    
    # Check output dimensions
    println(y)

    loss = AD.sum(y)
    println(loss)
    graph = AD.topological_sort(loss)

    AD.forward!(graph)
    AD.backward!(graph)

    @test size(y.output) == (embedding_dim, length(x_val))
    # Test gradient for W exists and has correct shape
    @test !isnothing(embedding_layer.W.gradient)
    #@test size(embedding_layer.W.gradient) == size(embedding_layer.W.output)

    # Test no gradient propagates to indices (x)
    @test isnothing(x.gradient)  # x is integer indices - should have no gradient

    # Verify loss is a scalar
end