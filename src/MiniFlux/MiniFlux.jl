module MiniFlux

using LinearAlgebra

# Ładujemy AutoDiff z folderu AutoDiff (przyjmujemy, że AutoDiff znajduje się w include path)
using ..AutoDiff
const AD = AutoDiff


# Ładujemy podmoduły biblioteki MiniFlux
include("layers.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")

# Eksportujemy publiczny interfejs
export Dense, mse_loss, sgd!, train!, Model, σ, relu, tanh


# Struktura modelu – sekwencja warstw oraz lista parametrów, które będą aktualizowane
struct Model
    layers::Vector   # Pozwalamy na dowolne wywoływalne obiekty
    params::Vector{AD.Variable}
end


# Forward modelu: przepuszcza wejście przez wszystkie warstwy
function (m::Model)(x)
    a = x
    for layer in m.layers
        a = layer(a)
    end
    return a
end

end # module MiniFlux
