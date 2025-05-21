function train!(model, loss_fn, train_data, val_data, opt, epochs::Int; lr=0.01)

    opt_fn = opt isa DataType ? opt() : opt

    for epoch in 1:epochs
        total_loss = 0.0
        total_samples = 0

        for (x_val, y_val) in train_data
            for p in model.params
                p.gradient = zeros(size(p.output))
            end

            x = AD.Variable(x_val, name="x")
            y = AD.Variable(y_val, name="y")
            ŷ = model(x)
            loss = loss_fn(y, ŷ)

            graph = AD.topological_sort(loss)
            AD.forward!(graph)
            AD.backward!(graph)
            opt_fn(model.params, lr)

            total_loss += loss.output
            total_samples += size(y_val, 2)
        end

        X_train = hcat([b[1] for b in train_data]...)
        Y_train = hcat([b[2] for b in train_data]...)
        train_acc = accuracy(model, X_train, Y_train)

        X_val = hcat([b[1] for b in val_data]...)
        Y_val = hcat([b[2] for b in val_data]...)
        val_acc = accuracy(model, X_val, Y_val)

        avg_loss = total_loss / total_samples
        println("Epoch $epoch: loss=$(round(avg_loss,digits=4)), train_acc=$(round(train_acc*100,digits=2))%, val_acc=$(round(val_acc*100,digits=2))%")
    end
end



function create_batches(X, y, batchsize)
    n = size(X, 2)
    batches = []
    for i in 1:batchsize:n
        batch_end = min(i + batchsize - 1, n)
        xs = X[:, i:batch_end]
        ys = y[:, i:batch_end]
        push!(batches, (xs, ys))
    end
    return batches
end

function accuracy(model, X::AbstractMatrix, Y::AbstractMatrix)
    # forward
    x_var   = AD.Variable(X, name="x")
    y_pred  = model(x_var)
    AD.forward!(AD.topological_sort(y_pred))
    ŷ       = y_pred.output

    C, N = size(ŷ)
    if C == 1
        # binary
        probs  = vec(ŷ)
        truths = vec(Y)
        preds  = probs .> 0.5
        truths = truths .> 0.5
        return sum(preds .== truths) / N
    else
        # multiclass
        pred_classes = [argmax(view(ŷ, :, i)) for i in 1:N]
        true_classes = [argmax(view(Y, :, i)) for i in 1:N]
        return sum(pred_classes .== true_classes) / N
    end
end