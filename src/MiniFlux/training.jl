# W MiniFlux/training.jl

# Upewnij się, że Optimisers jest dostępne w tym pliku, jeśli nie jest w głównym module
# Jeśli MiniFlux jest modułem, możesz potrzebować:
# using Optimisers # Jeśli Optimisers jest w głównym środowisku
# lub jeśli training.jl jest częścią modułu, który re-eksportuje Optimisers
# Jeśli nie, dodaj `using Optimisers` na początku pliku training.jl
# W MiniFlux/training.jl
# Upewnij się, że masz `using Optimisers` i `using Printf` (dla @sprintf) na początku pliku,
# jeśli nie są one globalnie dostępne z modułu MiniFlux.
# using Printf # Jeśli jeszcze nie ma

function train_with_optimisers!(
    model::Model,
    loss_fn::Function,
    train_data::AbstractVector,
    val_data::AbstractVector,
    opt_rule, # Np. Optimisers.Adam(0.001)
    epochs::Int
)
    println("<<<<< Entered train_with_optimisers! function >>>>>"); flush(stdout)
    
    # Przygotowanie stanu optymalizatora
    # Zbieramy 'surowe' tablice parametrów, które będą aktualizowane
    # To jest ważne, bo Optimisers.update zwróci nowe tablice, które musimy skopiować z powrotem
    initial_trainable_params_arrays = [p.output for p in model.params]
    opt_state = Optimisers.setup(opt_rule, initial_trainable_params_arrays)
    println("train_with_optimisers!: Optimizer state created."); flush(stdout)

    num_train_batches = length(train_data)

    for epoch in 1:epochs
        epoch_time_start = time()
        println("===== Starting Epoch $epoch/$epochs ====="); flush(stdout)
        
        total_loss_epoch = 0.0
        total_samples_epoch = 0
        
        # Bieżące tablice parametrów do aktualizacji przez Optimisers.jl
        # Zaczynamy od tych z modelu, aktualizujemy je co batch
        current_trainable_params_arrays = [p.output for p in model.params]


        for (batch_idx, (x_val, y_val)) in enumerate(train_data)
            batch_time_start = time()
            
            # 1. Resetowanie gradientów
            for p in model.params
                # Bezpieczniejsze resetowanie, aby uniknąć problemów z typami lub `nothing`
                p.gradient = fill!(similar(p.output), 0) 
            end

            # 2. Forward i Backward pass (Twoja logika AutoDiff)
            x_node = AD.Variable(x_val, name="x")
            y_node = AD.Variable(y_val, name="y")
            ŷ_node = model(x_node) # Definicja grafu
            loss_node = loss_fn(y_node, ŷ_node) # Definicja grafu
            
            graph = AD.topological_sort(loss_node)
            AD.forward!(graph)  # Obliczenia
            AD.backward!(graph) # Obliczenia gradientów -> p.gradient są wypełnione

            # 3. Przygotowanie gradientów dla Optimisers.jl
            # grads_arrays powinien mieć taką samą strukturę jak current_trainable_params_arrays
            grads_arrays = Vector{Any}(undef, length(model.params))
            for i in 1:length(model.params)
                p_var = model.params[i]
                if p_var.gradient === nothing
                    # To nie powinno się zdarzyć, jeśli backward! działa poprawnie
                    println("Warning: Gradient for param $(p_var.name) is nothing in batch $batch_idx, epoch $epoch.")
                    grads_arrays[i] = zeros(eltype(p_var.output), size(p_var.output))
                elseif size(p_var.gradient) != size(p_var.output)
                    # Jeśli kształty się nie zgadzają, próbujemy to naprawić lub rzucamy błąd
                    # Tutaj można dodać bardziej zaawansowaną logikę dopasowywania, jeśli jest potrzebna
                    # Na razie, dla bezpieczeństwa, logujemy i używamy zer, aby uniknąć błędu w Optimisers
                    println("Warning: Shape mismatch for param $(p_var.name). Param: $(size(p_var.output)), Grad: $(size(p_var.gradient)). Using zeros for this grad.")
                    grads_arrays[i] = zeros(eltype(p_var.output), size(p_var.output))
                else
                    grads_arrays[i] = p_var.gradient
                end
            end
            
            # 4. Aktualizacja parametrów za pomocą Optimisers.jl
            # Optimisers.update zwraca (nowy_stan, nowe_parametry_jako_tablice)
            new_opt_state, updated_params_arrays = Optimisers.update(opt_state, current_trainable_params_arrays, grads_arrays)
            
            # 5. Skopiowanie zaktualizowanych parametrów z powrotem do AD.Variable i przygotowanie na następny krok
            for (i, p_var) in enumerate(model.params)
                p_var.output .= updated_params_arrays[i]
                current_trainable_params_arrays[i] = updated_params_arrays[i] # Ważne dla następnego wywołania Optimisers.update
            end
            opt_state = new_opt_state # Zapisz nowy stan optymalizatora

            # Akumulacja statystyk
            batch_loss = loss_node.output
            total_loss_epoch += batch_loss
            total_samples_epoch += size(y_val, 2) # Zakładając, że drugi wymiar y_val to rozmiar batcha

            batch_time_end = time()
            batch_duration = batch_time_end - batch_time_start

            # Logowanie postępu co określoną liczbę batchy lub na końcu
            # np. co 10% batchy lub co 50 batchy
            print_interval = max(1, div(num_train_batches, 10))
            if batch_idx % print_interval == 0 || batch_idx == num_train_batches
                @printf("Epoch %d, Batch %d/%d: Loss: %.4f, Time/Batch: %.3fs\n",
                        epoch, batch_idx, num_train_batches, batch_loss / size(y_val,2), batch_duration)
                flush(stdout)
            end
        end # Koniec pętli po batchach

        epoch_time_end = time()
        epoch_duration = epoch_time_end - epoch_time_start
        avg_batch_time = epoch_duration / num_train_batches
        avg_loss_epoch = total_loss_epoch / total_samples_epoch

        # Ewaluacja na zbiorze walidacyjnym i treningowym (agregowanym)
        println("Epoch $epoch: Calculating accuracies..."); flush(stdout)
        # Zbieranie danych do ewaluacji (może być kosztowne, jeśli zbiory są duże)
        # Można to robić rzadziej lub na mniejszej próbce.
        X_train_agg = hcat([b[1] for b in train_data]...); Y_train_agg = hcat([b[2] for b in train_data]...)
        train_acc = accuracy(model, X_train_agg, Y_train_agg)
        
        X_val_agg = hcat([b[1] for b in val_data]...); Y_val_agg = hcat([b[2] for b in val_data]...)
        val_acc = accuracy(model, X_val_agg, Y_val_agg)

        @printf("EPOCH %d SUMMARY: Avg Loss: %.4f, Train Acc: %.2f%%, Val Acc: %.2f%%, Epoch Time: %.2fs (Avg Batch: %.3fs)\n",
                epoch, avg_loss_epoch, train_acc*100, val_acc*100, epoch_duration, avg_batch_time)
        println("====================================="); flush(stdout)

    end # Koniec pętli po epokach
    println("<<<<< train_with_optimisers! function finished >>>>>"); flush(stdout)
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