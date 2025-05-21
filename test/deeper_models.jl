using Test
using Random
using myExample

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "MiniFlux Iris Test with Batching" begin
    # Load data
    include("iris.jl")  # zakładam, że definiuje: inputs, targets

    # Shuffle and split
    iris_data = [(vec(inputs[i, :]), vec(targets[i, :])) for i in 1:size(inputs, 1)]
    Random.seed!(42)
    shuffle!(iris_data)

    n = length(iris_data)
    n_train = Int(floor(0.7 * n))

    # Create train/test sets as matrices
    train_inputs  = hcat(first.(iris_data[1:n_train])...)  # (features, n_train)
    train_targets = hcat(last.(iris_data[1:n_train])...)   # (outputs,  n_train)

    test_inputs  = hcat(first.(iris_data[n_train+1:end])...)
    test_targets = hcat(last.(iris_data[n_train+1:end])...)

    # Create batches
    batch_size = 16
    train_batches = MF.create_batches(train_inputs, train_targets, batch_size)
    test_batches = MF.create_batches(test_inputs, test_targets, batch_size)

    # Build model
    model = MF.chain(
        MF.Dense(4, 5, AD.relu, bias=true),
        MF.Dense(5, 3, AD.softmax, bias=true)
    )

    # Train
    MF.train!(model, MF.categorical_cross_entropy, train_batches, test_batches, MF.sgd!, 5; lr=0.01)

    # Accuracy function for matrix input
    function multiclass_accuracy(model, X::Matrix, Y::Matrix)
    # 1) Forward
        x_var  = AD.Variable(X, name="x")
        y_pred = model(x_var)
        AD.forward!(AD.topological_sort(y_pred))
        ŷ = y_pred.output            # (C, N)

        # 2) Predykcje i prawda
        # a) predykcje: numer klasy o największym logitcie
        pred_classes = [argmax(view(ŷ, :, i)) for i in 1:size(ŷ, 2)]
        # b) prawdziwe: indeks jedynki w one-hot Y
        true_classes = [argmax(view(Y, :, i)) for i in 1:size(Y, 2)]

        return sum(pred_classes .== true_classes) / length(pred_classes)
    end



    acc_train = multiclass_accuracy(model, train_inputs, train_targets)
    acc_test  = multiclass_accuracy(model, test_inputs, test_targets)

    println("Accuracy on training set: ", round(acc_train * 100, digits=2), "%")
    println("Accuracy on test set: ", round(acc_test * 100, digits=2), "%")

    @test acc_train > 0.7
    @test acc_test  > 0.6
end
