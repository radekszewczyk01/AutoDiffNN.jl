using Test
using myExample
using Random

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "Dense layer with different activations" begin
    # Test with vector input
    x_val = randn(3)
    x = AD.Variable(x_val, name="x")
    
    activations = [AD.relu, AD.linear, AD.σ, AD.swish]
    activation_names = ["ReLU", "Linear", "Sigmoid", "Swish"]
    
    for (activation, name) in zip(activations, activation_names)
        @testset "$name activation" begin
            layer = MF.Dense(3, 2, activation)
            y = layer(x)
            loss = AD.sum(y)
            
            graph = AD.topological_sort(loss)
            loss_val = AD.forward!(graph)
            AD.backward!(graph)
            
            @test isa(loss_val, Number)
            @test !isnothing(x.gradient)
            @test !isnothing(layer.W.gradient)
            @test !isnothing(layer.b.gradient)
        end
    end
    
    # Test with matrix input (batch)
    x_batch = randn(3, 4)  # 3 features, 4 samples
    x = AD.Variable(x_batch, name="x")
    
    for (activation, name) in zip(activations, activation_names)
        @testset "$name activation with batch" begin
            layer = MF.Dense(3, 2, activation)
            y = layer(x)
            loss = AD.sum(y)
            
            graph = AD.topological_sort(loss)
            loss_val = AD.forward!(graph)
            AD.backward!(graph)
            
            @test isa(loss_val, Number)
            @test !isnothing(x.gradient)
            @test !isnothing(layer.W.gradient)
            @test !isnothing(layer.b.gradient)
            @test size(y.output) == (2, 4)  # Check output shape
        end
    end
end

@testset "BatchNorm layer" begin
    x_val = randn(3, 4)  # 3 features, 4 samples
    x = AD.Variable(x_val, name="x")
    
    bn = MF.BatchNorm(3)
    y = bn(x)
    loss = AD.sum(y)
    
    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)
    
    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test !isnothing(bn.gamma.gradient)
    @test !isnothing(bn.beta.gradient)
    @test size(y.output) == size(x_val)
end

@testset "Dropout layer" begin
    x_val = randn(3, 4)
    x = AD.Variable(x_val, name="x")
    
    dropout = MF.Dropout(0.5)
    y = dropout(x)
    loss = AD.sum(y)
    
    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)
    
    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test size(y.output) == size(x_val)
end

@testset "Sequential model" begin
    x_val = randn(3, 4)
    x = AD.Variable(x_val, name="x")
    
    model = MF.Sequential(
        MF.Dense(3, 4, AD.relu),
        MF.BatchNorm(4),
        MF.Dropout(0.3),
        MF.Dense(4, 2, AD.linear)
    )
    
    y = model(x)
    loss = AD.sum(y)
    
    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)
    
    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test size(y.output) == (2, 4)
    
    # Test parameter collection
    params = MF.get_parameters(model)
    @test length(params) > 0
    @test all(p -> !isnothing(p.gradient), params)
end
