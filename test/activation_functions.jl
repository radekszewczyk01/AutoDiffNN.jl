using Test
using myExample
using Random

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "Dense layer with AD.relu activation" begin
    # Wejście: wektor długości 3
    x_val = randn(3)
    x = AD.Variable(x_val, name="x")

    # Warstwa gęsta 3 -> 2, aktywacja: ReLU
    layer = MF.Dense(3, 2, AD.relu; bias=true)

    # Forward: wynik przed aktywacją i po aktywacji
    y = layer(x)

    # Funkcja celu: suma wyjść (skalarny cel)
    loss = AD.sum(y)

    # Przepływ w przód i wstecz
    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)

    # Sprawdzenie: czy wynik i gradienty są poprawne
    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test all(.!isnothing.(layer.W.gradient))
    if layer.b !== nothing
        @test layer.b === nothing || all(.!isnothing.(layer.b.gradient))
    end
end

@testset "Dense layer with AD.linear activation" begin
    x_val = randn(3)
    x = AD.Variable(x_val, name="x")

    layer = MF.Dense(3, 2, AD.linear; bias=true)

    y = layer(x)
    loss = AD.sum(y)

    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)

    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test all(.!isnothing.(layer.W.gradient))
    if layer.b !== nothing
        @test layer.b === nothing || all(.!isnothing.(layer.b.gradient))
    end
end

@testset "Dense layer with AD.sigmoid (σ) activation" begin
    x_val = randn(3)
    x = AD.Variable(x_val, name="x")

    layer = MF.Dense(3, 2, AD.σ; bias=true)

    y = layer(x)
    loss = AD.sum(y)

    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)

    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test all(.!isnothing.(layer.W.gradient))
    if layer.b !== nothing
        @test layer.b === nothing || all(.!isnothing.(layer.b.gradient))
    end
end

@testset "Dense layer with AD.swish activation" begin
    x_val = randn(3)
    x = AD.Variable(x_val, name="x")

    layer = MF.Dense(3, 2, AD.swish; bias=true)

    y = layer(x)
    loss = AD.sum(y)

    graph = AD.topological_sort(loss)
    loss_val = AD.forward!(graph)
    AD.backward!(graph)

    @test isa(loss_val, Number)
    @test !isnothing(x.gradient)
    @test all(.!isnothing.(layer.W.gradient))
    if layer.b !== nothing
        @test layer.b === nothing || all(.!isnothing.(layer.b.gradient))
    end
end
