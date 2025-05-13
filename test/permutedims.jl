using Test
using Random
using myExample

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "PermuteDims Operator Tests" begin
    Random.seed!(1234)
    x = rand(2,3,4)
    order = (3,2,1)

    # zbuduj operator „na sucho”
    op = AD.PermuteDims(AD.Variable(x), order)

    # 1) Forward
    y = AD.forward(op, x)  # wywołuje nasz `forward(op, x)`
    @test y == permutedims(x, order)

    # 2) Backward
    grad = ones(size(y))
    dx = AD.backward(op, x, grad)[1]
    @test dx == permutedims(grad, invperm(order))
end
