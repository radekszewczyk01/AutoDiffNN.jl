using Test
using myExample
using Random

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "AutoDiff smoke test" begin
    x = AD.Variable(5.0, name="x")
    two = AD.Constant(2.0)

    squared = x^two
    sine = sin(squared)

    graph = AD.topological_sort(sine)
    y_val = AD.forward!(graph)
    AD.backward!(graph)

    expected_y = sin(25.0)
    expected_grad = 2 * 5.0 * cos(25.0)

    @test isapprox(y_val, expected_y; atol=1e-10)
    @test isapprox(x.gradient, expected_grad; atol=1e-10)
end

include("iris.jl")  

@testset "MiniFlux training test (Iris)" begin

    layer = MF.Dense(4, 3, AD.linear; bias=true)
    params = [layer.W, layer.b]
    model = MF.Model([layer], params)

    iris_data = [(vec(inputs[i, :]), vec(targets[i, :])) for i in 1:size(inputs, 1)]

    x_before = AD.Variable(vec(inputs[1, :]), name="x")
    y_true   = AD.Variable(vec(targets[1, :]), name="y")
    ŷ_before = model(x_before)
    loss_before = MF.mse_loss(y_true, ŷ_before)
    graph_before = AD.topological_sort(loss_before)
    initial_loss = AD.forward!(graph_before)

    # 1 epoch with small learning rate
    MF.train!(model, MF.mse_loss, iris_data, MF.sgd!, 1; lr=0.001)

    # loss after trainig
    x_after = AD.Variable(vec(inputs[1, :]), name="x")
    y_true_after   = AD.Variable(vec(targets[1, :]), name="y")
    ŷ_after = model(x_after)
    loss_after = MF.mse_loss(y_true_after, ŷ_after)
    graph_after = AD.topological_sort(loss_after)
    new_loss = AD.forward!(graph_after)

    println("Initial loss: ", initial_loss)
    println("New loss after training: ", new_loss)

    @test new_loss < initial_loss 
end

function accuracy(model, data)
    correct = 0
    for (x_val, y_val) in data

        x_var = AD.Variable(x_val, name="x")
        y_pred_var = model(x_var)
        
        y_out = AD.forward!(AD.topological_sort(y_pred_var))

        pred = argmax(y_out)
        truth = argmax(y_val)
        correct += (pred == truth)
    end
    return correct / length(data)
end

@testset "MiniFlux Iris – train/test split" begin
    include("iris.jl")

    iris_data = [(vec(inputs[i, :]), vec(targets[i, :])) for i in 1:size(inputs,1)]

    Random.seed!(42)
    shuffle!(iris_data)

    n = length(iris_data)
    n_train = Int(floor(0.7n))
    train_data = iris_data[1:n_train]
    test_data  = iris_data[n_train+1:end]

    layer = MF.Dense(4, 3, AD.swish; bias=true)
    model = MF.Model([layer], [layer.W, layer.b])

    MF.train!(model, MF.mse_loss, train_data, MF.sgd!, 10; lr=0.01)

    acc_train = accuracy(model, train_data)
    acc_test  = accuracy(model, test_data)

    println("Accuracy on training set: ", round(acc_train*100, digits=2), "%")
    println("Accuracy on  test   set: ", round(acc_test*100,  digits=2), "%")

    @test acc_train > 0.7    
    @test acc_test  > 0.6   
end
