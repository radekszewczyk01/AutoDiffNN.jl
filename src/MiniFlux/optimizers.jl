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

mutable struct Adam
    m::Dict{UInt,Any}
    v::Dict{UInt,Any}
    β1::Float64
    β2::Float64
    ϵ::Float64
    t::Int
    function Adam(; β1=0.9, β2=0.999, ϵ=1e-8)
        inst = new(Dict{UInt,Any}(), Dict{UInt,Any}(), β1, β2, ϵ, 0)
        return inst
    end
end

function (opt::Adam)(params::Vector{AD.Variable}, lr)
    println("Dummy Adam update called. opt.t before: $(opt.t), num_params: $(length(params)), lr: $lr"); flush(stdout)
    opt.t += 1
    return nothing
end