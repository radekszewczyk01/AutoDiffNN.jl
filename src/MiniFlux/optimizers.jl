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
        new(Dict(), Dict(), β1, β2, ϵ, 0)
    end
end
function (opt::Adam)(params::Vector{Variable}, lr)
    opt.t += 1
    for p in params
        g = p.gradient
        # Convert gradient to match parameter size
        if size(g) != size(p.output)
            # If total elements match, reshape
            if length(g) == length(p.output)
                g = reshape(g, size(p.output))
            # Handle vector/matrix conversion
            elseif length(g) == length(p.output) && ndims(g) != ndims(p.output)
                g = reshape(g, size(p.output))
            # Handle (1,1) vs (1,) conversion
            elseif prod(size(g)) == length(p.output)
                g = reshape(g, size(p.output))
            else
                error("""
                Gradient size $(size(g)) doesn't match parameter size $(size(p.output)) 
                for parameter '$(p.name)'. Total elements: $(length(g)) vs $(length(p.output)).
                """)
            end
        end
        @assert size(g) == size(p.output) "Size mismatch: $(size(g)) != $(size(p.output)) for '$(p.name)'"
        
        key = objectid(p)
        m = get!(opt.m, key, zero(g))
        v = get!(opt.v, key, zero(g))

        m .= opt.β1 .* m .+ (1 .- opt.β1) .* g
        v .= opt.β2 .* v .+ (1 .- opt.β2) .* (g .^ 2)

        m̂ = m ./ (1 .- opt.β1^opt.t)
        v̂ = v ./ (1 .- opt.β2^opt.t)
        p.output .-= lr .* m̂ ./ (sqrt.(v̂) .+ opt.ϵ)
    end
end