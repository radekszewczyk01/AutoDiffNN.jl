using Test
using Random
using myExample
using Optimisers

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

@testset "MiniFlux Iris Test with Batching and Optimisers.jl" begin
    include("iris.jl")  
    iris_data = [(vec(inputs[i, :]), vec(targets[i, :])) for i in 1:size(inputs, 1)]
    Random.seed!(42)
    shuffle!(iris_data)

    n = length(iris_data)
    n_train = Int(floor(0.7 * n))

    train_inputs  = hcat(first.(iris_data[1:n_train])...)  # (features, n_train)
    train_targets = hcat(last.(iris_data[1:n_train])...)   # (outputs,  n_train)

    test_inputs  = hcat(first.(iris_data[n_train+1:end])...)
    test_targets = hcat(last.(iris_data[n_train+1:end])...)

    batch_size = 16
    train_batches = MF.create_batches(train_inputs, train_targets, batch_size)
    test_batches = MF.create_batches(test_inputs, test_targets, batch_size)

    model = MF.chain(
        MF.Dense(4, 10, AD.relu),
        MF.Dense(10, 3, AD.softmax)
    )

    println("Model built. Structure:")
    for (i, layer) in enumerate(model.layers)
        println("  Layer $i: $layer")
    end

    learning_rate = 0.01f0
    opt_rule = Optimisers.Adam(learning_rate)

    println("Starting training...")

    MF.train_with_optimisers!(
        model,
        MF.categorical_cross_entropy,
        train_batches,
        test_batches,
        opt_rule,
        5
    )
    println("Training finished.")

    function multiclass_accuracy(model_eval, X::Matrix, Y::Matrix)
        x_var  = AD.Variable(X, name="x_eval")
        y_pred_node = model_eval(x_var)
        
        eval_graph = AD.topological_sort(y_pred_node)
        AD.forward!(eval_graph)
        
        ŷ = y_pred_node.output

        if ŷ === nothing
            error("Model output is nothing during accuracy calculation. Forward pass might have failed or was not run.")
        end
        if !(ndims(ŷ) == 2 && size(ŷ,1) == size(Y,1) && size(ŷ,2) == size(Y,2))
             error("Shape mismatch: ŷ has shape $(size(ŷ)), Y has shape $(size(Y))")
        end


        pred_classes = [argmax(view(ŷ, :, i)) for i in 1:size(ŷ, 2)]
        true_classes = [argmax(view(Y, :, i)) for i in 1:size(Y, 2)]

        return sum(pred_classes .== true_classes) / length(pred_classes)
    end

    acc_train = multiclass_accuracy(model, train_inputs, train_targets)
    acc_test  = multiclass_accuracy(model, test_inputs, test_targets)

    println("Accuracy on training set: ", round(acc_train * 100, digits=2), "%")
    println("Accuracy on test set: ", round(acc_test * 100, digits=2), "%")

    @test acc_train > 0.8
    @test acc_test  > 0.8
end
