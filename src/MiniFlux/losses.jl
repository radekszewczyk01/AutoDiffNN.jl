using LinearAlgebra

function mse_loss(y::AD.Variable, ŷ::AD.GraphNode)

    diff = y .- ŷ
    sq = diff .^ AD.Constant(2.0)    
    s = AD.sum(sq)                  

    return AD.Constant(0.5) * s        
end

function binary_cross_entropy_loss(y::AD.Variable, ŷ::AD.GraphNode)
    one = AD.Constant(1.0)
    # term1 = y * log(ŷ)
    t1 = AD.broadcasted(*, y, AD.broadcasted(log, ŷ))
    # term2 = (1−y) * log(1−ŷ)
    t2 = AD.broadcasted(*, AD.broadcasted(-, one, y),
                             AD.broadcasted(log, AD.broadcasted(-, one, ŷ)))
    # suma i odsunięcie minusa
    s = AD.broadcasted(+, t1, t2)
    loss_sum = AD.broadcasted(sum, s)          # to jest scala wszystkie elementy
    # mnożenie przez -1/N
    N = length(y.output)
    return AD.broadcasted(*, AD.Constant(-1/N), loss_sum)
end

