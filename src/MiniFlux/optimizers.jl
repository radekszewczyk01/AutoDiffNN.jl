using LinearAlgebra


function sgd!(params::Vector{AD.Variable}, lr::Float64)
    for p in params
        p.output .-= lr .* p.gradient

        p.gradient = zeros(size(p.output))
    end
end

