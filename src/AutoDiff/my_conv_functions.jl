function my_conv(X::AbstractArray{T,3}, K::AbstractArray{T,3}; stride::Int=1, padding::Int=0) where T
    seq_len, in_channels_X, batch_size = size(X)
    kernel_width, in_channels_K, out_channels = size(K)

    if in_channels_X != in_channels_K
        error("Liczba kanałów wejściowych w X ($(in_channels_X)) musi być równa liczbie kanałów wejściowych w K ($(in_channels_K)).")
    end

    if padding != 0
        if padding == div(kernel_width - 1, 2) && stride == 1
            X_padded = similar(X, (seq_len + 2*padding, in_channels_X, batch_size))
            fill!(X_padded, zero(T))
            X_padded[padding+1:padding+seq_len, :, :] = X
            X = X_padded
            seq_len = size(X,1)
        else
            @warn "Pełna obsługa paddingu nie jest jeszcze zaimplementowana. Używam padding=0, jeśli nie jest to 'same' padding dla stride=1."
            padding = 0 
        end
    end

    out_seq_len = div(seq_len - kernel_width + 2*padding, stride) + 1

    if out_seq_len <= 0
        error("Kernel jest za duży dla danych wejściowych z tym paddingiem i stride. out_seq_len = $out_seq_len")
    end

    Y = zeros(T, out_seq_len, out_channels, batch_size)

    for n in 1:batch_size
        for c_out in 1:out_channels
            for w_out in 1:out_seq_len
                w_in_start_orig = (w_out - 1) * stride + 1
                acc = zero(T)
                for c_in in 1:in_channels_X
                    for kw in 1:kernel_width

                        w_in_idx = (w_out - 1) * stride - padding + kw 

                        if padding == 0
                            idx_in_X = (w_out - 1) * stride + kw
                            if 1 <= idx_in_X <= seq_len
                                acc += X[idx_in_X, c_in, n] * K[kw, c_in, c_out]
                            end
                        else
                            idx_in_X_padded = (w_out - 1) * stride + kw
                            if 1 <= idx_in_X_padded <= seq_len
                                acc += X[idx_in_X_padded, c_in, n] * K[kw, c_in, c_out]
                            end
                        end
                    end
                end
                Y[w_out, c_out, n] = acc
            end
        end
    end
    return Y
end

function ∇my_conv_data(g_incoming::AbstractArray{T,3}, K_val::AbstractArray{T,3};
                             X_shape::Tuple{Int,Int,Int}, stride::Int=1, padding::Int=0) where T
    out_seq_len_g, out_channels_g, batch_size = size(g_incoming)
    kernel_width, in_channels_K, out_channels_K = size(K_val)

    if out_channels_g != out_channels_K
        error("Liczba kanałów wyjściowych w g_incoming ($(out_channels_g)) musi być równa liczbie kanałów wyjściowych w K ($(out_channels_K)).")
    end

    seq_len_X_orig, in_channels_X, _ = X_shape
    if in_channels_X != in_channels_K
        error("Liczba kanałów wejściowych w X_shape ($(in_channels_X)) musi być równa liczbie kanałów wejściowych w K ($(in_channels_K)).")
    end

    dX = zeros(T, seq_len_X_orig, in_channels_X, batch_size)

    if padding == 0
        for n in 1:batch_size
            for c_in in 1:in_channels_X
                for w_out_g in 1:out_seq_len_g
                    for c_out in 1:out_channels_K
                        grad_val = g_incoming[w_out_g, c_out, n]
                        for kw in 1:kernel_width
                            idx_x_seq = (w_out_g - 1) * stride + kw
                            if 1 <= idx_x_seq <= seq_len_X_orig
                                dX[idx_x_seq, c_in, n] += grad_val * K_val[kw, c_in, c_out]
                            end
                        end
                    end
                end
            end
        end
    else
        for n in 1:batch_size
            for c_in in 1:in_channels_X
                for w_out_g in 1:out_seq_len_g
                    for c_out in 1:out_channels_K
                        grad_val = g_incoming[w_out_g, c_out, n]
                        for kw in 1:kernel_width
                            idx_x_orig = (w_out_g - 1) * stride + kw - padding
                            
                            if 1 <= idx_x_orig <= seq_len_X_orig
                                dX[idx_x_orig, c_in, n] += grad_val * K_val[kw, c_in, c_out]
                            end
                        end
                    end
                end
            end
        end
    end
    return dX
end

function ∇my_conv_filter(X_orig::AbstractArray{T,3}, g_incoming::AbstractArray{T,3};
                               K_shape::Tuple{Int,Int,Int}, stride::Int=1, padding::Int=0) where T
    seq_len_X_orig, in_channels_X, batch_size = size(X_orig)
    out_seq_len_g, out_channels_g, _ = size(g_incoming)

    kernel_width_K, in_channels_K, out_channels_K = K_shape

    if in_channels_X != in_channels_K
        error("Liczba kanałów wejściowych w X ($(in_channels_X)) musi być równa liczbie kanałów wejściowych w K_shape ($(in_channels_K)).")
    end
    if out_channels_g != out_channels_K
        error("Liczba kanałów wyjściowych w g_incoming ($(out_channels_g)) musi być równa liczbie kanałów wyjściowych w K_shape ($(out_channels_K)).")
    end

    X_eff = X_orig
    seq_len_X_eff = seq_len_X_orig

    if padding != 0
         if padding == div(kernel_width_K - 1, 2) && stride == 1
            X_padded = similar(X_orig, (seq_len_X_orig + 2*padding, in_channels_X, batch_size))
            fill!(X_padded, zero(T))
            X_padded[padding+1:padding+seq_len_X_orig, :, :] = X_orig
            X_eff = X_padded
            seq_len_X_eff = size(X_eff,1)
        else
            @warn "Pełna obsługa paddingu nie jest jeszcze zaimplementowana dla ∇my_conv_filter. Używam padding=0 dla X."
        end
    end

    dK = zeros(T, kernel_width_K, in_channels_K, out_channels_K)

    for n in 1:batch_size
        for c_out in 1:out_channels_K
            for c_in in 1:in_channels_K
                for kw in 1:kernel_width_K
                    acc = zero(T)
                    for w_out_g in 1:out_seq_len_g
                        idx_x_eff = (w_out_g - 1) * stride + kw
                        if 1 <= idx_x_eff <= seq_len_X_eff
                            acc += X_eff[idx_x_eff, c_in, n] * g_incoming[w_out_g, c_out, n]
                        end
                    end
                    dK[kw, c_in, c_out] += acc
                end
            end
        end
    end
    return dK
end

function my_maxpool(X::AbstractArray{T,3}, pool_width::Int, stride_val::Int; padding_val::Int=0) where T
    if padding_val != 0
        error("my_maxpool obsługuje obecnie tylko padding_val = 0.")
    end

    seq_len_X, num_channels, batch_size = size(X)

    out_seq_len = fld(seq_len_X - pool_width, stride_val) + 1

    if out_seq_len <= 0
        error("Okno poolingu ($pool_width) jest za duże dla danych wejściowych ($seq_len_X) z danym krokiem ($stride_val). Długość wyjściowa <= 0.")
    end

    Y = zeros(T, out_seq_len, num_channels, batch_size)

    for n in 1:batch_size
        for c in 1:num_channels
            for w_out in 1:out_seq_len

                start_idx_X = (w_out - 1) * stride_val + 1
                end_idx_X = start_idx_X + pool_width - 1

                window = X[start_idx_X:end_idx_X, c, n]
                Y[w_out, c, n] = maximum(window)
            end
        end
    end
    return Y
end

function ∇my_maxpool(g_incoming::AbstractArray{Tg,3}, 
                     Y_cache::AbstractArray{Ty,3}, 
                     X_cache::AbstractArray{Tx,3}, 
                     pool_width::Int, 
                     stride_val::Int; 
                     padding_val::Int=0) where {Tg, Ty, Tx}
    
    if padding_val != 0
        error("∇my_maxpool obsługuje obecnie tylko padding_val = 0.")
    end

    seq_len_X, num_channels, batch_size = size(X_cache)
    out_seq_len_Y, _, _ = size(Y_cache)

    dX = zeros(promote_type(Tg, Ty, Tx), seq_len_X, num_channels, batch_size)

    for n in 1:batch_size
        for c in 1:num_channels
            for w_out_Y in 1:out_seq_len_Y
                grad_val = g_incoming[w_out_Y, c, n]

                if grad_val == zero(Tg) 
                    continue
                end

                max_val_from_Y = Y_cache[w_out_Y, c, n]

                start_idx_X_window = (w_out_Y - 1) * stride_val + 1
                end_idx_X_window = start_idx_X_window + pool_width - 1

                found_max_in_window = false
                for current_x_idx_in_window in start_idx_X_window:end_idx_X_window
                    if X_cache[current_x_idx_in_window, c, n] == max_val_from_Y
                        dX[current_x_idx_in_window, c, n] += grad_val
                        found_max_in_window = true
                        break
                    end
                end
            end
        end
    end
    return dX
end