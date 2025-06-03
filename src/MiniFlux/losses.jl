using LinearAlgebra

function mse_loss(y, ŷ)
    return AD.sum(AD.Constant(0.5) .* (y .- ŷ).^AD.Constant(2))
end

function categorical_cross_entropy(y::AD.GraphNode, ŷ::AD.GraphNode)
    return AD.Constant(-1) * AD.sum(y .* AD.broadcasted(log, ŷ))
end

function binary_cross_entropy_loss(y_target::AD.GraphNode, y_pred::AD.GraphNode; ϵ=1f-7) # Float32 epsilon
    one = AD.Constant(1.0f0)

    log_ŷ = AD.broadcasted(log, y_pred)
    log_1_minus_ŷ = AD.broadcasted(log, AD.broadcasted(-, one, y_pred))


    term1 = y_target .* log_ŷ
    term2 = (one .- y_target) .* log_1_minus_ŷ
    
    sum_terms = AD.sum(term1 .+ term2)

    return AD.Constant(-1.0f0) .* sum_terms
end


function softmax(x::AD.GraphNode)
    ex  = AD.broadcasted(exp, x)
    s   = AD.sum(ex)
    return ex / s
end


