using LinearAlgebra
function mse_loss(y::AD.Variable, ŷ::AD.GraphNode)
    # oblicz różnicę jako wektor w grafie
    diff = y .- ŷ                       
    # kwadrat element-wise w grafie
    sq   = diff .^ AD.Constant(2.0)    
    # sumuj wszystkie składowe (tworzy skalar w grafie)
    s    = AD.sum(sq)                  
    # pomnóż przez 0.5 w grafie i zwróć skalar
    return AD.Constant(0.5) * s        
end

