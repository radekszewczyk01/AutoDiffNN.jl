import Base: ^, sin, *, sum, max
import LinearAlgebra: mul!
import Base.Broadcast: broadcasted

# Dodana funkcja unbroadcast_gradient
function unbroadcast_gradient(grad_output::AbstractArray, input_shape::Tuple)
    # Jeśli wejście było skalarem, input_shape to ().
    # Gradientem dla skalarnego wejścia jest suma wszystkich elementów grad_output.
    if isempty(input_shape)
        return sum(grad_output)
    end

    if size(grad_output) == input_shape
        return grad_output
    end

    # Sumuj wzdłuż wymiarów, które były broadcastowane
    sum_dims = Int[]
    for i in 1:ndims(grad_output)
        s_out = size(grad_output, i)
        s_in = (i <= length(input_shape)) ? input_shape[i] : 1 # Rozmiar wejścia w tym wymiarze
        
        if s_out > 1 && s_in == 1
            push!(sum_dims, i)
        end
    end
    
    grad_in = grad_output
    if !isempty(sum_dims)
        grad_in = sum(grad_in, dims=Tuple(sum_dims))
    end
    
    return reshape(grad_in, input_shape)
end

function unbroadcast_gradient(grad_output::Number, input_shape::Tuple)
    if isempty(input_shape) # Wejście było skalarem, grad_output jest skalarem.
        return grad_output
    end
    # Jeśli grad_output jest skalarem, a input_shape nie jest puste,
    # to oznacza, że skalar (grad_output) nie wymaga dalszej redukcji wymiarów.
    # To może się zdarzyć, jeśli np. grad_output_raw był macierzą, ale input_shape było (),
    # wtedy sum(grad_output_raw) dało skalar, który jest tu przekazany.
    # Lub jeśli sama operacja na skalarach dała skalar.
    return grad_output
end


^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

# Mnożenie macierzy - nie jest to operacja broadcastowana element-wise w typowym sensie
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x) # Użycie BroadcastedOperator tutaj może być mylące, ale zachowuję strukturę
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = begin
    # println("Forward for mul!: typeof(A)=$(typeof(A)), eltype(A)=$(eltype(A)), typeof(x)=$(typeof(x)), eltype(x)=$(eltype(x))"); flush(stdout)
    res = A * x
    # println("Forward for mul!: typeof(res=A*x)=$(typeof(res)), eltype(res)=$(eltype(res))"); flush(stdout)
    return res
end
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g) # Standardowe gradienty dla mnożenia macierzy, kształty powinny być ok

broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x_val, y_val) = x_val .* y_val
backward(node::BroadcastedOperator{typeof(*)}, x_val, y_val, g) = (
    unbroadcast_gradient(g .* y_val, size(x_val)),
    unbroadcast_gradient(g .* x_val, size(y_val))
)

broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x_val, y_val) = x_val .- y_val
backward(::BroadcastedOperator{typeof(-)}, x_val, y_val, g) = (
    unbroadcast_gradient(g, size(x_val)),
    unbroadcast_gradient(-g, size(y_val))
)

broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x_val, y_val) = begin
    # println("Forward for +: typeof(x_val)=$(typeof(x_val)), eltype(x_val)=$(eltype(x_val)), typeof(y_val)=$(typeof(y_val)), eltype(y_val)=$(eltype(y_val))"); flush(stdout)
    res = x_val .+ y_val
    # println("Forward for +: typeof(res=x_val.+y_val)=$(typeof(res)), eltype(res)=$(eltype(res))"); flush(stdout)
    return res
end
backward(::BroadcastedOperator{typeof(+)}, x_val, y_val, g) = (
    unbroadcast_gradient(g, size(x_val)),
    unbroadcast_gradient(g, size(y_val))
)

broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x_val, y_val) = x_val ./ y_val
backward(node::BroadcastedOperator{typeof(/)}, x_val, y_val, g) = (
  unbroadcast_gradient(g ./ y_val, size(x_val)),
  unbroadcast_gradient(-g .* x_val ./ (y_val .^ 2), size(y_val))
)

σ(x::GraphNode) = BroadcastedOperator(σ, x) # Zmieniono na GraphNode, aby było spójne
forward(::BroadcastedOperator{typeof(σ)}, x_val) = return 1.0 ./ (1.0 .+ exp.(-x_val))
backward(node::BroadcastedOperator{typeof(σ)}, x_val, g) = begin # Zmieniono na begin/end dla wielu linii
    # println("Backward for σ (sigmoid), typeof(g): $(typeof(g)), eltype(g): $(eltype(g))"); flush(stdout)
    # println("Backward for σ (sigmoid), typeof(x_val): $(typeof(x_val)), eltype(x_val): $(eltype(x_val))"); flush(stdout)
    
    y = node.output # y = σ(x_val), powinno być Float32 jeśli x_val jest Float32 i forward używa 1.0f0
    # println("Backward for σ (sigmoid), typeof(y = node.output): $(typeof(y)), eltype(y): $(eltype(y))"); flush(stdout)

    # Sprawdźmy typy przed operacją .*
    # g, y, (1-y)
    # Jeśli g jest Float64, a y jest Float32, to g .* y będzie Float64
    one_val = eltype(y)(1) # Tworzymy jedynkę tego samego typu co y
    
    grad_x = g .* y .* (one_val .- y)
    # println("Backward for σ (sigmoid), typeof(grad_x = g .* y .* (1 .- y)): $(typeof(grad_x)), eltype(grad_x): $(eltype(grad_x))"); flush(stdout)
    
    # unbroadcast_gradient powinien zachować typ grad_x
    final_grad = unbroadcast_gradient(grad_x, size(x_val))
    # println("Backward for σ (sigmoid), typeof(final_grad after unbroadcast): $(typeof(final_grad)), eltype(final_grad): $(eltype(final_grad))"); flush(stdout)
    
    return (final_grad,)
end
Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x_val, y_val) = return x_val .^ y_val
backward(node::BroadcastedOperator{typeof(^)}, x_val, y_val, g) = let
    # Należy uważać na dziedzinę log, x_val musi być > 0 jeśli y_val nie jest całkowite
    # log(abs(x_val)) jest bezpieczniejsze, jeśli dopuszczamy ujemne x_val
    gx_raw = g .* y_val .* x_val .^ (y_val .- 1)
    gy_raw = g .* log.(abs.(x_val)) .* (x_val .^ y_val) # log.(abs.(x)) to standard
    return (
        unbroadcast_gradient(gx_raw, size(x_val)),
        unbroadcast_gradient(gy_raw, size(y_val))
    )
end

Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x_val) = return exp.(x_val)
backward(node::BroadcastedOperator{typeof(exp)}, x_val, g) = let
    y = node.output # exp(x_val)
    grad_x = g .* y
    return (unbroadcast_gradient(grad_x, size(x_val)),)
end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x_val) = return log.(x_val) # x_val musi być dodatnie
backward(::BroadcastedOperator{typeof(log)}, x_val, g) = (
    unbroadcast_gradient(g .* (1.0 ./ x_val), size(x_val)),
)

# sum jest operacją redukującą, unbroadcast_gradient nie jest tu bezpośrednio stosowany w ten sam sposób
sum(x::GraphNode) = BroadcastedOperator(sum, x) # Użycie BroadcastedOperator dla sum jest kwestią konwencji
forward(::BroadcastedOperator{typeof(sum)}, x_val) = sum(x_val)
backward(node::BroadcastedOperator{typeof(sum)}, x_val, g) = (
  fill(g, size(x_val)), # Gradient dla sumy to g rozgłoszone do kształtu wejścia
)

broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x_val, y_val) = max.(x_val, y_val)
backward(node::BroadcastedOperator{typeof(max)}, x_val, y_val, g) = (
  unbroadcast_gradient(g .* (x_val .>= y_val), size(x_val)),
  unbroadcast_gradient(g .* (y_val .>  x_val), size(y_val)) # y .> x, a nie y .>= x, aby uniknąć podwójnego liczenia gradientu gdy x==y
)

linear(x::GraphNode) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x_val) = x_val
backward(::BroadcastedOperator{typeof(linear)}, x_val, g) = (
  unbroadcast_gradient(g, size(x_val)),
)

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x_val) = max.(x_val, 0.0f0)
backward(node::BroadcastedOperator{typeof(relu)}, x_val, g) = (
  unbroadcast_gradient(g .* (x_val .> 0.0f0), size(x_val)),
)

swish(x::GraphNode) = BroadcastedOperator(swish, x)
forward(::BroadcastedOperator{typeof(swish)}, x_val) = x_val ./ (1 .+ exp.(-x_val))
backward(node::BroadcastedOperator{typeof(swish)}, x_val, g) = let
    # deriv = σ(x) + x * σ(x) * (1-σ(x))
    # y = node.output (swish(x_val))
    # σx = 1.0 ./ (1.0 .+ exp.(-x_val))
    # deriv_swish = σx .+ x_val .* σx .* (1.0 .- σx)
    # LUB: y = x*σ(x) => dy/dx = σ(x) + x*σ'(x) = σ(x) + x*σ(x)(1-σ(x))
    # LUB: swish(x) = x / (1+exp(-x)) = x * sigmoid(x)
    # swish_prime(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #                = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    #                = sigmoid(x) * (1 + x - x * sigmoid(x))
    #                = node.output/x * (1 + x - node.output) (jeśli x != 0)
    #                = (node.output + x * sigmoid(x) * (1-sigmoid(x))) -> jak wyżej
    σx_val = 1.0 ./ (1.0 .+ exp.(-x_val))
    deriv = σx_val .+ x_val .* σx_val .* (1.0 .- σx_val)
    grad_x = g .* deriv
    return (unbroadcast_gradient(grad_x, size(x_val)),)
end

# softmax jest bardziej złożony i jego gradient ma specyficzną formę
softmax(x::GraphNode) = BroadcastedOperator(softmax, x; name="softmax")
forward(::BroadcastedOperator{typeof(softmax)}, x_val) = begin
    # x_val może być wektorem (N,) lub macierzą (N, Batch)
    # softmax liczymy po pierwszym wymiarze (features)
    exp_x = exp.(x_val .- maximum(x_val, dims=1)) # Stabilizacja numeryczna
    sum_exp_x = sum(exp_x; dims=1)
    return exp_x ./ sum_exp_x
end
backward(actual_node_object::BroadcastedOperator{typeof(softmax)}, x_val_from_fwd, g) = begin
    # println("Backward for softmax, typeof(g)=$(typeof(g)), typeof(x_val_from_fwd)=$(typeof(x_val_from_fwd))"); flush(stdout)
    y = actual_node_object.output # Użyj nazwanego argumentu
    # println("Backward for softmax, typeof(y=node.output)=$(typeof(y))"); flush(stdout)
    s = sum(g .* y; dims=1)
    grad_x = y .* (g .- s)
    # println("Backward for softmax, typeof(grad_x)=$(typeof(grad_x))"); flush(stdout)
    return (grad_x,) # Zwracamy krotkę, bo softmax jest unarny
end

mutable struct EmbeddingOp <: Operator
    inputs::Any # (W_node, x_indices_node)
    output::Any
    gradient::Any
    name::String
    indices_matrix_cache::Any # Cache indices for backward pass
    EmbeddingOp(W_node::GraphNode, x_indices_node::GraphNode; name="embedding") = 
        new((W_node, x_indices_node), nothing, nothing, name, nothing)
end
# Funkcja pomocnicza, aby można było pisać AD.EmbeddingOp(...) w MiniFlux
EmbeddingOp(W::GraphNode, x_indices::GraphNode) = EmbeddingOp(W, x_indices, name="EmbeddingOp")


function forward(op::EmbeddingOp, W_matrix, x_indices_matrix)
    embed_dim, vocab_size = size(W_matrix)
    # x_indices_matrix może być (seq_len,) lub (seq_len, batch_size)
    if !(eltype(x_indices_matrix) <: Integer)
        error("Embedding input indices must be integers.")
    end

    op.indices_matrix_cache = x_indices_matrix # Cache for backward pass

    if ndims(x_indices_matrix) == 1 # Pojedyncza sekwencja
        seq_len = length(x_indices_matrix)
        batch_size = 1
        _indices = reshape(x_indices_matrix, seq_len, 1)
    elseif ndims(x_indices_matrix) == 2 # Batch sekwencji
        seq_len, batch_size = size(x_indices_matrix)
        _indices = x_indices_matrix
    else
        error("Embedding input must be 1D (seq_len) or 2D (seq_len, batch_size). Got $(ndims(x_indices_matrix))D")
    end
    
    out_tensor = similar(W_matrix, embed_dim, seq_len, batch_size)

    for b in 1:batch_size
        for s in 1:seq_len
            idx = _indices[s, b]
            if !(1 <= idx <= vocab_size)
                error("Index $idx out of vocab range 1:$vocab_size at seq pos $s, batch $b")
            end
            out_tensor[:, s, b] = W_matrix[:, idx]
        end
    end
    # Jeśli oryginalne wejście było 1D, zwróć 2D (embed_dim, seq_len)
    return ndims(x_indices_matrix) == 1 ? out_tensor[:,:,1] : out_tensor
end

function backward(op::EmbeddingOp, W_matrix, _, g) # Drugi input (x_indices_matrix) nie jest używany, bierzemy z cache
    x_indices_matrix = op.indices_matrix_cache 
    embed_dim, vocab_size = size(W_matrix)
    
    if ndims(x_indices_matrix) == 1 # Pojedyncza sekwencja
        seq_len = length(x_indices_matrix)
        batch_size = 1
        _indices = reshape(x_indices_matrix, seq_len, 1)
        _g = ndims(g) == 2 ? reshape(g, embed_dim, seq_len, 1) : g # upewnij się, że g jest 3D
    elseif ndims(x_indices_matrix) == 2 # Batch sekwencji
        seq_len, batch_size = size(x_indices_matrix)
        _indices = x_indices_matrix
        _g = g
    else
        # To nie powinno się zdarzyć jeśli forward zadziałał
        error("Unexpected shape for x_indices_matrix in backward EmbeddingOp")
    end

    grad_W = zeros(eltype(W_matrix), size(W_matrix))

    for b in 1:batch_size
        for s in 1:seq_len
            idx = _indices[s, b]
            if 1 <= idx <= vocab_size
                grad_W[:, idx] .+= _g[:, s, b]
            end
        end
    end
    return (grad_W, nothing) # Brak gradientu dla dyskretnych indeksów
end
Base.show(io::IO, x::EmbeddingOp) = print(io, "op Embedding")

mutable struct PermuteDimsOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    dims_order::Tuple
    inv_dims_order::Tuple
    function PermuteDimsOp(x_node::GraphNode, dims_order::Tuple; name="permutedims")
        ido =- map(x -> findfirst(==(x), dims_order), 1:length(dims_order)) # szybka invperm
        # lub:
        inv_order = Vector{Int}(undef, length(dims_order))
        for (i, val) in enumerate(dims_order)
            inv_order[val] = i
        end
        new((x_node,), nothing, nothing, name, dims_order, tuple(inv_order...))
    end
end
PermuteDimsOp(x::GraphNode, dims::Tuple) = PermuteDimsOp(x, dims, name="PermuteDimsOp")

function forward(op::PermuteDimsOp, x_val)
    return permutedims(x_val, op.dims_order)
end

function backward(op::PermuteDimsOp, x_val, g)
    return (permutedims(g, op.inv_dims_order),)
end
Base.show(io::IO, x::PermuteDimsOp) = print(io, "op PermuteDims($(x.dims_order))")

mutable struct ReshapeOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    target_shape_dims::Union{Tuple, Nothing} # Może być NTuple{N, Int} lub Int...
    original_shape_cache::Any
    function ReshapeOp(x_node::GraphNode, target_shape_dims...; name="reshape") # akceptuje (Int,Int) lub Int,Int...
        new((x_node,), nothing, nothing, name, tuple(target_shape_dims...), nothing)
    end
end
ReshapeOp(x::GraphNode, ts::Tuple) = ReshapeOp(x, ts..., name="ReshapeOpToTuple")

function forward(op::ReshapeOp, x_val)
    op.original_shape_cache = size(x_val)
    return reshape(x_val, op.target_shape_dims...)
end

function backward(op::ReshapeOp, x_val, g)
    return (reshape(g, op.original_shape_cache),)
end
Base.show(io::IO, x::ReshapeOp) = print(io, "op Reshape($(x.target_shape_dims))")

# FlattenOp
mutable struct FlattenOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    original_shape_cache::Any
    function FlattenOp(x_node::GraphNode; name="flatten")
        new((x_node,), nothing, nothing, name, nothing)
    end
end

function forward(op::FlattenOp, x_val)
    op.original_shape_cache = size(x_val)
    if ndims(x_val) == 1 # Już jest "płaski" (wektor)
        return reshape(x_val, (length(x_val), 1)) # Zawsze zwracaj (features, 1) dla pojedynczego przykładu
    elseif ndims(x_val) == 2 && size(x_val,2)==1 # Już (features,1)
        return x_val
    end

    # Zakładamy, że ostatni wymiar to batch_size
    batch_size = size(x_val)[end]
    num_features = div(length(x_val), batch_size)
    return reshape(x_val, (num_features, batch_size))
end

function backward(op::FlattenOp, x_val, g)
    # println("Backward for FlattenOp, typeof(g): $(typeof(g))"); flush(stdout)

    return (reshape(g, op.original_shape_cache),)
end
Base.show(io::IO, x::FlattenOp) = print(io, "op Flatten")

mutable struct Conv1DOp <: Operator
    inputs::Any # (x_node, kernel_node)
    output::Any
    gradient::Any
    name::String
    # Cache dla backward pass
    X_val_cache::Any
    K_val_cache::Any
    Conv1DOp(x_node::GraphNode, kernel_node::GraphNode; name="conv1d") = 
        new((x_node, kernel_node), nothing, nothing, name, nothing, nothing)
end
Conv1DOp(x::GraphNode, k::GraphNode) = Conv1DOp(x, k, name="Conv1DOp")
# Twoja obecna konwencja:
# X_val: (Win, Cin, B) - zgadza się z NNlib (W, C_in, N)
# K_val: (KW, Cin_k, Cout) - zgadza się z NNlib (FW, C_in, C_out)
# Y: (Wout, Cout, B) - zgadza się z NNlib (W_out, C_out, N)



mutable struct MaxPool1DOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    pool_size::Tuple{Int}
    # Cache dla backward pass
    X_val_cache::Any 
    Y_val_cache::Any # Cache wyjścia Y jest potrzebny, aby znaleźć max indices w backward
    # lub max_indices_mask jak poprzednio, ale wymaga ostrożności z mapowaniem
    MaxPool1DOp(x_node::GraphNode, pool_size::Tuple{Int}; name="maxpool1d") = 
        new((x_node,), nothing, nothing, name, pool_size, nothing, nothing)
end
MaxPool1DOp(x::GraphNode, ps::Tuple{Int}) = MaxPool1DOp(x, ps, name="MaxPool1DOp")

# W forward dla Conv1DOp
function forward(op::Conv1DOp, X_val::AbstractArray{T,3}, K_val::AbstractArray{T,3}) where T
    op.X_val_cache = X_val
    op.K_val_cache = K_val

    # Dla konwolucji 1D z stride=1, padding=0 ('valid'), dilation=1
    # NNlib może oczekiwać, że stride, padding, dilation będą tuplami o długości
    # równej liczbie wymiarów przestrzennych konwolucji (czyli 1 dla 1D).
    cdims = DenseConvDims(X_val, K_val; stride=1, padding=0, dilation=1) # Zmieniono na wartości skalarne dla 1D
    # Alternatywnie, jeśli oczekuje tupli:
    # cdims = DenseConvDims(X_val, K_val; stride=(1,), padding=(0,), dilation=(1,))
    
    return NNlib.conv(X_val, K_val, cdims)
end

function backward(op::Conv1DOp, X_val_z_fwd::AbstractArray{Tx,3}, K_val_z_fwd::AbstractArray{Tk,3}, g_incoming::AbstractArray{Tg,3}) where {Tx, Tk, Tg}
    # println("Backward for Conv1DOp, typeof(g_incoming): $(typeof(g_incoming)), eltype: $(eltype(g_incoming))"); flush(stdout)
    
    # Używamy wartości z cache, bo X_val_z_fwd i K_val_z_fwd to te same wartości
    X_val_f32 = op.X_val_cache::Array{Float32,3}
    K_val_f32 = op.K_val_cache::Array{Float32,3}

    G_val_f32 = g_incoming
    if eltype(g_incoming) != Float32
        # println("Conv1DOp backward: CONVERTING g_incoming from $(eltype(g_incoming)) to Float32"); flush(stdout)
        G_val_f32 = convert(Array{Float32,3}, g_incoming)
    end
    # ... reszta jak poprzednio ...
    cdims = DenseConvDims(X_val_f32, K_val_f32; stride=1, padding=0, dilation=1)
    dX = NNlib.∇conv_data(G_val_f32, K_val_f32, cdims)
    dK = NNlib.∇conv_filter(X_val_f32, G_val_f32, cdims)
    # println("Conv1DOp backward: typeof(dX)=$(typeof(dX)), typeof(dK)=$(typeof(dK))"); flush(stdout)
    return (dX, dK) # Krotka z dwoma gradientami (dla X i dla K)
end

# ==============================================================================
# MaxPool1DOp z użyciem NNlib
# ==============================================================================
function forward(op::MaxPool1DOp, X_val::AbstractArray{T,3}) where T
    op.X_val_cache = X_val

    # Win, C, B = size(X_val) # Nieużywane
    pool_width = op.pool_size[1]
    
    pdims = NNlib.PoolDims(X_val, (pool_width,); stride=(pool_width,), padding=(0,))
    
    Y = NNlib.maxpool(X_val, pdims)
    op.Y_val_cache = Y 
    return Y
end


function backward(op::MaxPool1DOp, x_val_z_forward::AbstractArray{Tx,3}, g_incoming::AbstractArray{Tg,3}) where {Tx, Tg}
    # println("Backward for MaxPool1DOp, typeof(g_incoming): $(typeof(g_incoming)), eltype: $(eltype(g_incoming))"); flush(stdout)
    
    # X_val_z_forward to wejście do MaxPool z forward pass. Możemy go użyć lub X_val_cache.
    # Lepiej używać cache, jeśli jest spójne.
    X_val_f32 = op.X_val_cache::Array{Float32,3} # Cache jest Float32
    Y_val_f32 = op.Y_val_cache::Array{Float32,3} # Cache jest Float32

    G_val_f32 = g_incoming # Używamy gradientu "z góry"
    if eltype(g_incoming) != Float32
        # println("MaxPool1DOp backward: CONVERTING g_incoming from $(eltype(g_incoming)) to Float32"); flush(stdout)
        G_val_f32 = convert(Array{Float32,3}, g_incoming)
    end
    # println("MaxPool1DOp backward: Using G_val_f32 type: $(typeof(G_val_f32)), eltype: $(eltype(G_val_f32))"); flush(stdout)

    pool_width = op.pool_size[1]
    pdims = NNlib.PoolDims(X_val_f32, (pool_width,); stride=(pool_width,), padding=(0,))
    
    dX = NNlib.∇maxpool(G_val_f32, Y_val_f32, X_val_f32, pdims)
    # println("MaxPool1DOp backward: typeof(dX)=$(typeof(dX))"); flush(stdout)
    
    return (dX,) # Zwracamy krotkę, bo backward! oczekuje kolekcji gradientów dla wejść
end