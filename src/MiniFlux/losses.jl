using LinearAlgebra

function mse_loss(y, ŷ)
    return AD.sum(AD.Constant(0.5) .* (y .- ŷ).^AD.Constant(2))
end

function categorical_cross_entropy(y::AD.GraphNode, ŷ::AD.GraphNode)
    return AD.Constant(-1) * AD.sum(y .* AD.broadcasted(log, ŷ))
end

function binary_cross_entropy_loss(y::AD.Variable, ŷ::AD.GraphNode)
    one = AD.Constant(1.0)
    t1 = AD.broadcasted(*, y, AD.broadcasted(log, ŷ))
    t2 = AD.broadcasted(*, AD.broadcasted(-, one, y),
                             AD.broadcasted(log, AD.broadcasted(-, one, ŷ)))
    s = AD.broadcasted(+, t1, t2)
    loss_sum = AD.broadcasted(sum, s)
    N = length(y.output)
    return AD.broadcasted(*, AD.Constant(-1/N), loss_sum)
end


function softmax(x::AD.GraphNode)
    ex  = AD.broadcasted(exp, x)
    s   = AD.sum(ex)
    return ex / s
end


