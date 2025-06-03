using LinearAlgebra

function sgd!(params::Vector{Variable}, lr)
    for p in params
        g = p.gradient
        if size(g) != size(p.output)
            g = sum(g, dims=2)
            g = vec(g)
        end
        @assert size(g) == size(p.output)
        p.output .-= lr .* g
    end
end

# W MiniFlux/optimizers.jl
mutable struct Adam
    m::Dict{UInt,Any} # Odkomentuj
    v::Dict{UInt,Any} # Odkomentuj
    β1::Float64
    β2::Float64
    ϵ::Float64
    t::Int
    function Adam(; β1=0.9, β2=0.999, ϵ=1e-8)
        # Usunięte println dla czystości, możesz je zostawić jeśli chcesz
        # println("Inside Adam constructor about to call new()..."); flush(stdout)
        inst = new(Dict{UInt,Any}(), Dict{UInt,Any}(), β1, β2, ϵ, 0) # Przywróć Dict()
        # println("Adam instance created with new()..."); flush(stdout)
        return inst
    end
end

# Funkcja (opt::Adam)(...) pozostaje bez zmian (z użyciem opt.m i opt.v)
# ... struct Adam i jej konstruktor (pełna wersja ze słownikami) ...

# TYMCZASOWO ZAKOMENTUJ ORYGINALNĄ IMPLEMENTACJĘ:
# function (opt::Adam)(params::Vector{AD.Variable}, lr)
#     opt.t += 1
#     # ... cała skomplikowana logika ...
# end

# WSTAW BARDZO PROSTĄ IMPLEMENTACJĘ NA CZAS TESTU:
function (opt::Adam)(params::Vector{AD.Variable}, lr) # Upewnij się, że AD.Variable jest poprawne
    println("Dummy Adam update called. opt.t before: $(opt.t), num_params: $(length(params)), lr: $lr"); flush(stdout)
    opt.t += 1 # Jakaś minimalna operacja na obiekcie opt
    # Nie rób nic z params na razie
    # Możesz nawet dodać return nothing
    return nothing
end