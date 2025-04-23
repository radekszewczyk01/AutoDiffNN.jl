using LinearAlgebra

# Funkcje aktywacji
σ(x) = 1 ./ (1 .+ exp.(-x))    # funkcja sigmoidalna
relu(x) = max.(0, x)
tanh(x) = Base.tanh.(x)

# Warstwa Dense – wykorzystuje AutoDiff do budowy grafu obliczeniowego
struct Dense
    W::AD.Variable
    b::Union{AD.Variable, Nothing}
    activation::Function
end

# Konstruktor warstwy Dense. Parametr bias jest opcjonalny.
function Dense(in_dim::Int, out_dim::Int, activation::Function=identity; bias::Bool=true)
    W = AD.Variable(randn(out_dim, in_dim), name="W")
    b = bias ? AD.Variable(randn(out_dim), name="b") : nothing
    return Dense(W, b, activation)
end

# Metoda forward dla warstwy Dense
function (layer::Dense)(x::AD.Variable)
    # 1) stwórz node operatora * pomiędzy W i x
    lin = layer.W * x               # ← to wywoła AD-owy Operator(*)
    # 2) dodaj bias (również AD-owy broadcast-Operator)
    z   = layer.b === nothing ? lin : lin .+ layer.b  
    # 3) nie wyciągaj z .output — zwróć bezpośrednio AD.Variable z grafem
    #    (jeśli activation to identity, po prostu z; inaczej aktywuj AD-owo)
    return layer.activation === identity ? z : layer.activation(z)
end
