using Test
using Random
using myExample

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "MaxPool1DOperator (AutoDiff)" begin
    Random.seed!(42)
    x = rand(2, 8, 3)
    pool = 4
    op = AD.MaxPool1D(AD.Constant(x), pool)
    y = AD.forward(op, x)
    @test size(y) == (2, 2, 3)

    expected = zeros(2, 2, 3)
    for b in 1:2, c in 1:3, i in 1:2
        segment = x[b, (i-1)*pool+1 : i*pool, c]
        expected[b,i,c] = maximum(segment)
    end
    @test y == expected

    grad = ones(size(y))
    dx = AD.backward(op, x, grad)[1]
    for b in 1:2, c in 1:3
        for i in 1:2, j in 1:pool
            pos = (i-1)*pool + j
            @test dx[b,pos,c] in (0, 1.0)
        end
    end
end

@testset "MaxPool1D Layer (MiniFlux)" begin
    x = rand(2, 8, 3)
    x_node = AD.Variable(x)
    layer = MF.MaxPool1D(4)
    y_node = layer(x_node)

    g = AD.topological_sort(y_node)
    AD.forward!(g)
    @test size(y_node.output) == (2, 2, 3)

    loss = AD.sum(y_node)
    graph = AD.topological_sort(loss)
    AD.forward!(graph)
    AD.backward!(graph)
    @test size(x_node.gradient) == size(x)
end
