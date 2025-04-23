using LinearAlgebra

# Aktualizacja parametrów przy użyciu SGD
function sgd!(params::Vector{AD.Variable}, lr::Float64)
    for p in params
        @assert !isnothing(p.gradient) "Gradient nie został obliczony dla parametru $(p.name)"
        p.output .-= lr .* p.gradient
        # Zerujemy gradient – inicjujemy macierz zer o tym samym rozmiarze co p.output
        p.gradient = zeros(size(p.output))
    end
end

