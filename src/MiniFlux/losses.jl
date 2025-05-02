using LinearAlgebra

function mse_loss(y::AD.Variable, ŷ::AD.GraphNode)

    # roznica jako wektor w grafie
    diff = y .- ŷ

    # kwadrat element-wise w grafie
    sq = diff .^ AD.Constant(2.0)    
    s = AD.sum(sq)                  

    return AD.Constant(0.5) * s        
end

