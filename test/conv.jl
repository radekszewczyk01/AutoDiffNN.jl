using Test
using Random
using myExample

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "Conv1D layer forward and backward pass" begin
    batch_size = 2
    seq_len = 10
    in_channels = 3
    out_channels = 8
    kernel_size = (3,)
    x_val = randn(batch_size, seq_len, in_channels)
    x = AD.Variable(x_val, name="x")

    conv_layer = MF.Conv(kernel_size, in_channels => out_channels, AD.relu)

    y = conv_layer(x)
    
    expected_seq_len = seq_len - kernel_size[1] + 1

    loss = AD.sum(y)

    graph = AD.topological_sort(loss)
    AD.forward!(graph)
    AD.backward!(graph)
    @test size(y.output) == (batch_size, expected_seq_len, out_channels)

    @test isa(loss.output, Number)
    @test !isnothing(conv_layer.W.gradient)
    @test size(conv_layer.W.gradient) == size(conv_layer.W.output)
    @test !isnothing(x.gradient)
    @test size(x.gradient) == size(x.output)
    
    if conv_layer.b !== nothing
        @test !isnothing(conv_layer.b.gradient)
        @test size(conv_layer.b.gradient) == size(conv_layer.b.output)
    end
end