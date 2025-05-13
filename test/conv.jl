using Test
using Random
using myExample  # Załaduj bibliotekę zawierającą AutoDiff i MiniFlux

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "Conv1D layer forward and backward pass" begin
    # Test parameters
    batch_size = 2
    seq_len = 10
    in_channels = 3
    out_channels = 8
    kernel_size = (3,)
    x_val = randn(batch_size, seq_len, in_channels)  # (batch, sequence, channels)
    x = AD.Variable(x_val, name="x")

    # Create Conv1D layer (3 kernel size, 8 output channels)
    conv_layer = MF.Conv(kernel_size, in_channels => out_channels, AD.relu)

    # Forward pass
    y = conv_layer(x)
    
    # Check output dimensions
    expected_seq_len = seq_len - kernel_size[1] + 1  # For stride=1, pad=0

    loss = AD.sum(y)

    graph = AD.topological_sort(loss)
    AD.forward!(graph)
    AD.backward!(graph)
    @test size(y.output) == (batch_size, expected_seq_len, out_channels)

    # Test gradients
    @test isa(loss.output, Number)  # Loss should be scalar
    @test !isnothing(conv_layer.W.gradient)  # Weight gradient exists
    @test size(conv_layer.W.gradient) == size(conv_layer.W.output)
    @test !isnothing(x.gradient)  # Input gradient exists
    @test size(x.gradient) == size(x.output)
    
    # Test bias gradient if present
    if conv_layer.b !== nothing
        @test !isnothing(conv_layer.b.gradient)
        @test size(conv_layer.b.gradient) == size(conv_layer.b.output)
    end
end