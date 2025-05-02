using LinearAlgebra

function mse_loss(y::AD.Variable, ŷ::AD.GraphNode)

    diff = y .- ŷ
    sq = diff .^ AD.Constant(2.0)    
    s = AD.sum(sq)                  

    return AD.Constant(0.5) * s        
end

