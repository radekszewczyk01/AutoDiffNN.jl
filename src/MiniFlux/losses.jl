using LinearAlgebra

function mse_loss(y, ŷ)
    return AD.sum(AD.Constant(0.5) .* (y .- ŷ).^AD.Constant(2))
end

function categorical_cross_entropy(y::AD.GraphNode, ŷ::AD.GraphNode)
    return AD.Constant(-1) * AD.sum(y .* AD.broadcasted(log, ŷ))
end

# w MiniFlux/losses.jl
function binary_cross_entropy_loss(y_target::AD.GraphNode, y_pred::AD.GraphNode; ϵ=1f-7) # Float32 epsilon
    # y_target.output i y_pred.output powinny być (1, batch_size) lub (batch_size,)
    # Jeśli y_target to AD.Variable, a y_pred to GraphNode z modelu
    one = AD.Constant(1.0f0)
    epsilon_node = AD.Constant(ϵ)

    # ŷ_clipped = max.(min.(ŷ, 1-ϵ), ϵ) - nie mamy max/min jako GraphNode operacji z trzema argumentami
    # Prostsze: ŷ + ϵ wewnątrz log
    # log_ŷ = AD.broadcasted(log, y_pred .+ epsilon_node) # Dodaj epsilon do argumentu log
    # log_1_minus_ŷ = AD.broadcasted(log, one .- y_pred .+ epsilon_node) # Dodaj epsilon

    # Standardowa forma bez epsilona (zakładając, że AutoDiff/log.jl poradzi sobie):
    log_ŷ = AD.broadcasted(log, y_pred)
    log_1_minus_ŷ = AD.broadcasted(log, AD.broadcasted(-, one, y_pred))


    term1 = y_target .* log_ŷ
    term2 = (one .- y_target) .* log_1_minus_ŷ
    
    sum_terms = AD.sum(term1 .+ term2) # Suma po wszystkich elementach batcha
    
    # Liczba przykładów w batchu
    # N = length(y_target.output) - to nie zadziała w momencie budowy grafu
    # Załóżmy, że y_target.output ma (1, batch_size) lub (features, batch_size)
    # Jeśli loss ma być skalarem (średnia na batch), N to batch_size
    # Jeśli y_target.output to np. macierz (1, N), to N = size(y_target.output,2)
    # Trudno to uzyskać dynamicznie w AutoDiff bez .output
    # Można przekazać N jako stałą lub obliczyć później.
    # Dla uproszczenia, na razie -sum(...). Uśrednianie można zrobić poza AD lub dodać operację Mean.
    # Flux.binarycrossentropy domyślnie robi `mean`.
    # Jeśli chcesz `mean`, musisz podzielić przez liczbę elementów.
    # N = AD.Constant(Float32(length(y_target.output))) # To zadziała tylko jeśli y_target to Constant z konkretnym outputem
    # Jeśli y_target jest Variable, jego .output będzie znane dopiero w forward!
    # Lepiej dzielić przez stałą wartość batch_size jeśli jest znana, lub zaimplementować mean operator.
    # Na razie, pomińmy dzielenie przez N - to tylko skalar mnożący gradienty.
    # Lub, jeśli N jest znane z góry:
    # N_val = Float32(size(y_target.output, 2)) # jeśli y_target.output jest (features, batch_size)
    # return AD.Constant(-1.0f0 / N_val) .* sum_terms
    # Jeśli y_target i y_pred to (1, batch_size), to N = batch_size

    # Proponuję, aby loss zwracał sumę, a uśrednianie było robione w pętli treningowej
    # lub dodaj dzielenie przez stałą jeśli jest znana.
    # Flux zwraca mean. Twój `train!` sumuje `loss.output` i dzieli przez `total_samples`.
    # To jest OK, jeśli `loss.output` jest sumą dla batcha.
    
    # Utrzymajmy oryginalną formę, zwracając średnią stratę dla batcha
    # N = AD.Constant(1.0f0 / convert(Float32, length(y_target.output))) # to wymaga y_target by był Constant
    # Jeśli y_target to Variable, y_target.output jest Any.
    # Użyjmy -sum na razie, dzielenie w pętli treningowej.
    return AD.Constant(-1.0f0) .* sum_terms
end


function softmax(x::AD.GraphNode)
    ex  = AD.broadcasted(exp, x)
    s   = AD.sum(ex)
    return ex / s
end


