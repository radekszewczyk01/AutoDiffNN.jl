using Test
using myExample
using Random

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux


function accuracy(model, data)
    correct = 0
    for (x_val, y_val) in data

        # forward dla predykcji
        x_var = AD.Variable(x_val, name="x")
        y_pred_var = model(x_var)
        
        y_out = AD.forward!(AD.topological_sort(y_pred_var))

        # decyzja klasy
        pred = argmax(y_out)
        truth = argmax(y_val)
        correct += (pred == truth)
    end
    return correct / length(data)
end


@testset "MiniFlux Multi-Layer Model Test" begin
    include("iris.jl")

    iris_data = [(vec(inputs[i, :]), vec(targets[i, :])) for i in 1:size(inputs, 1)]

    Random.seed!(42)
    shuffle!(iris_data)

    # Split the data into training and testing sets
    n = length(iris_data)
    n_train = Int(floor(0.7 * n))
    train_data = iris_data[1:n_train]
    test_data  = iris_data[n_train+1:end]

    # Define a multi-layer model: 4 -> 5 -> 3 (with two hidden layers)
    layer1 = MF.Dense(4, 5, AD.relu, bias=true)
    layer2 = MF.Dense(5, 3, AD.swish, bias=true)
    model = MF.Model([layer1, layer2], [layer1.W, layer1.b, layer2.W, layer2.b])

    # # Train the model
    MF.train!(model, MF.mse_loss, train_data, MF.sgd!, 100; lr=0.003)

    # Evaluate accuracy
    acc_train = accuracy(model, train_data)
    acc_test  = accuracy(model, test_data)

    println("Accuracy on training set: ", round(acc_train * 100, digits=2), "%")
    println("Accuracy on test set: ", round(acc_test * 100, digits=2), "%")

    @test acc_train > 0.7    
    @test acc_test  > 0.6   
end
