using Test, Random
using myExample

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "FlattenOperator (AutoDiff)" begin
    Random.seed!(123)
    x = rand(2, 3, 4, 5)
    op = AD.Flatten(AD.Constant(x))
    
    y = AD.forward(op, x)
    @test size(y) == (2, 3*4*5)
    @test y == reshape(x, 2, 3*4*5)

    grad = rand(size(y)...)
    dx = AD.backward(op, x, grad)[1]
    @test size(dx) == size(x)
    @test dx == reshape(grad, size(x))
end

@testset "Flatten Layer (MiniFlux)" begin
    x = rand(2,3,4,5)
    x_node = AD.Variable(x)
    layer = MF.Flatten()
    y_node = layer(x_node)

    g = AD.topological_sort(y_node)
    AD.forward!(g)
    @test size(y_node.output) == (2, 60)

    loss = AD.sum(y_node)
    graph = AD.topological_sort(loss)
    AD.forward!(graph)
    AD.backward!(graph)
    @test size(x_node.gradient) == size(x)
end
