# test/conv_layers_tests.jl

using Test
using Random
using myExample
using NNlib # Potrzebne, jeśli będziemy bezpośrednio porównywać z NNlib lub do tworzenia danych

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

# --- Funkcje pomocnicze dla testów ---
function get_output_and_grads(op_node::AD.GraphNode, inputs_data::Tuple)
    # Tworzy GraphNode dla wejść
    input_nodes = [AD.Variable(data, name="input_$(i)") for (i, data) in enumerate(inputs_data)]
    
    # Zbuduj graf do testowanego operatora (lub warstwy)
    # To zależy, czy testujemy operator AD czy warstwę MF
    # Dla prostoty, załóżmy, że op_node to już wynik np. MF.Embedding(...) lub AD.Conv1DOp(...)
    # W tym przypadku op_node to już np. AD.Conv1DOp(input_nodes[1], input_nodes[2])
    # Ta funkcja musi być bardziej generyczna lub testy muszą budować graf

    # Na razie załóżmy, że `op_node` to węzeł wyjściowy operacji,
    # a `input_nodes` są jego bezpośrednimi wejściami (jeśli to operator)
    # lub `op_node` to wynik warstwy, a `input_nodes[1]` to jej wejście.

    # Uproszczenie: przekazujemy funkcję tworzącą węzeł operatora
    # op_constructor_fn(input_nodes...) -> op_node
    # Ta część wymaga przemyślenia, jak najlepiej testować pojedyncze operatory/warstwy.

    # Dla testowania warstwy MiniFlux:
    # layer = MF.Embedding(...)
    # output_node = layer(input_nodes[1])

    # Dla testowania operatora AutoDiff:
    # output_node = AD.Conv1DOp(input_nodes[1], input_nodes[2])

    # Powyższe jest zbyt skomplikowane dla ogólnej funkcji.
    # Zamiast tego, testy będą budować graf explicite.

    # Ta funkcja będzie używana do uruchomienia forward/backward na danym `output_node`
    # i zwrócenia jego wyjścia oraz gradientów jego wejść.
    
    # Reset gradientów wejść przed testem
    # for inp_node in input_nodes
    #     inp_node.gradient = nothing
    # end

    # Symulujemy funkcję straty skalarnej na wyjściu op_node
    # Aby to zrobić, sumujemy wszystkie elementy wyjścia op_node
    # (To uproszczenie, prawdziwa funkcja straty byłaby bardziej złożona)
    if ndims(op_node.output) > 0 && length(op_node.output) > 1 # Jeśli wyjście nie jest już skalarem
        loss_sim_node = AD.sum(op_node) # AD.sum tworzy BroadcastedOperator{typeof(sum)}
    else
        loss_sim_node = op_node # Jeśli wyjście jest już skalarem
    end

    graph = AD.topological_sort(loss_sim_node)
    
    # Forward pass
    AD.forward!(graph)
    output_val = op_node.output # Wyjście testowanego operatora/warstwy

    # Backward pass
    # Upewnij się, że gradienty wszystkich węzłów w grafie są zresetowane
    # (AD.forward! powinien to robić, ale dla pewności)
    for n in graph; if n isa AD.Variable || n isa AD.Operator; n.gradient = nothing; end; end

    AD.backward!(graph; seed=1.0f0) # Użyj seed Float32

    # Zbierz gradienty wejść do pierwotnego `op_node`
    # To jest skomplikowane, bo `op_node` może być głęboko w grafie do `loss_sim_node`
    # Potrzebujemy gradientów dla `input_nodes` które stworzyliśmy na początku.
    
    # Gradienty będą w polach .gradient oryginalnych `input_nodes`
    # które były przekazane do konstruktora op_node lub warstwy.
    # To zależy od tego, jak testy są skonstruowane.

    # Na razie ta funkcja zwróci tylko output_val. Gradienty będą sprawdzane w testach.
    return output_val
end


Random.seed!(123)

@testset "AutoDiff Convolutional Operators and MiniFlux Layers" begin

    @testset "AD.EmbeddingOp" begin
        vocab_size = 10
        embed_dim = 5
        seq_len = 3
        batch_size = 2

        W_val = randn(Float32, embed_dim, vocab_size)
        # x_indices może być 1D (pojedyncza sekwencja) lub 2D (batch)
        x_indices_batch = rand(1:vocab_size, seq_len, batch_size)
        x_indices_single = rand(1:vocab_size, seq_len)

        W_node = AD.Variable(W_val)
        
        # Test z batchem
        x_idx_node_batch = AD.Constant(x_indices_batch) # Użyj Constant, bo indeksy nie mają gradientu
        embed_op_node_batch = AD.EmbeddingOp(W_node, x_idx_node_batch)
        graph_batch = AD.topological_sort(embed_op_node_batch) # Testujemy sam operator
        AD.forward!(graph_batch)
        output_batch = embed_op_node_batch.output

        @test size(output_batch) == (embed_dim, seq_len, batch_size)
        @test eltype(output_batch) == Float32
        # Sprawdzenie wartości (proste)
        @test output_batch[:, 1, 1] ≈ W_val[:, x_indices_batch[1, 1]]

        # Backward pass (tylko kształt gradientu dla W)
        # Aby przetestować backward, potrzebujemy skalarnego wyjścia
        loss_node_batch = AD.sum(embed_op_node_batch) # Suma jako prosta funkcja straty
        graph_loss_batch = AD.topological_sort(loss_node_batch)
        # Resetuj gradienty przed backward
        W_node.gradient = nothing; embed_op_node_batch.gradient = nothing; loss_node_batch.gradient = nothing
        # (AD.forward! resetuje, ale dla pewności)
        AD.forward!(graph_loss_batch) # Potrzebne do ustawienia node.output dla sumy
        AD.backward!(graph_loss_batch; seed=1.0f0)
        
        @test W_node.gradient !== nothing
        @test size(W_node.gradient) == size(W_val)
        @test eltype(W_node.gradient) == Float32

        # Test z pojedynczą sekwencją
        x_idx_node_single = AD.Constant(x_indices_single)
        embed_op_node_single = AD.EmbeddingOp(W_node, x_idx_node_single)
        graph_single = AD.topological_sort(embed_op_node_single)
        AD.forward!(graph_single)
        output_single = embed_op_node_single.output
        @test size(output_single) == (embed_dim, seq_len) # Bez wymiaru batcha
        @test eltype(output_single) == Float32
    end

    @testset "MF.Embedding Layer" begin
        vocab_size = 10
        embed_dim = 5
        seq_len = 3
        batch_size = 2
        
        layer = MF.Embedding(vocab_size, embed_dim)
        @test layer.W.output isa Matrix{Float32}
        @test size(layer.W.output) == (embed_dim, vocab_size)

        x_indices_val = rand(1:vocab_size, seq_len, batch_size)
        x_input_node = AD.Variable(x_indices_val) # Indeksy jako Variable, choć gradient nie będzie liczony
        
        output_node = layer(x_input_node) # To tworzy AD.EmbeddingOp
        
        # Uruchomienie forward pass
        # Trzeba stworzyć graf aż do output_node
        # W tym przypadku output_node jest już węzłem operatora
        graph = AD.topological_sort(output_node)
        AD.forward!(graph)
        
        @test output_node.output isa Array{Float32, 3}
        @test size(output_node.output) == (embed_dim, seq_len, batch_size)
    end

    @testset "AD.PermuteDimsOp and MF.Permute Layer" begin
        data_3d = randn(Float32, 10, 5, 2) # W, C, B
        data_node = AD.Variable(data_3d)
        
        # Test operatora
        perm_op_node = AD.PermuteDimsOp(data_node, (2,1,3)) # C, W, B
        graph_op = AD.topological_sort(perm_op_node)
        AD.forward!(graph_op)
        @test size(perm_op_node.output) == (5, 10, 2)
        @test eltype(perm_op_node.output) == Float32

        # Test warstwy
        perm_layer = MF.Permute((2,1,3))
        output_node_layer = perm_layer(data_node) # Tworzy ten sam PermuteDimsOp
        graph_layer = AD.topological_sort(output_node_layer)
        # AD.forward! już wykonany dla data_node, ale perm_op_node mógł nie być w tym grafie
        # Bezpieczniej jest zawsze robić forward na grafie do testowanego węzła
        AD.forward!(graph_layer) # Uruchom dla grafu warstwy
        @test size(output_node_layer.output) == (5, 10, 2)
        @test eltype(output_node_layer.output) == Float32

        # Test backward (kształt gradientu)
        loss_node = AD.sum(output_node_layer)
        graph_loss = AD.topological_sort(loss_node)
        data_node.gradient = nothing; output_node_layer.gradient = nothing; loss_node.gradient = nothing;
        AD.forward!(graph_loss)
        AD.backward!(graph_loss; seed=1.0f0)
        @test data_node.gradient !== nothing
        @test size(data_node.gradient) == size(data_3d)
        @test eltype(data_node.gradient) == Float32
    end

    @testset "AD.Conv1DOp and MF.Conv1D Layer (with NNlib)" begin
        W_in, C_in, B = 10, 3, 2  # Szerokość, Kanały wej, Batch
        KW, C_out = 3, 5          # Szerokość Kernela, Kanały wyj
        
        X_val = randn(Float32, W_in, C_in, B)
        K_val = randn(Float32, KW, C_in, C_out) # (KW, Cin, Cout)
        
        X_node = AD.Variable(X_val)
        K_node = AD.Variable(K_val)

        # Test operatora
        conv_op_node = AD.Conv1DOp(X_node, K_node)
        graph_op = AD.topological_sort(conv_op_node)
        AD.forward!(graph_op)
        W_out_expected = W_in - KW + 1
        @test size(conv_op_node.output) == (W_out_expected, C_out, B)
        @test eltype(conv_op_node.output) == Float32

        # Test backward (kształty gradientów)
        loss_node_op = AD.sum(conv_op_node)
        graph_loss_op = AD.topological_sort(loss_node_op)
        X_node.gradient = nothing; K_node.gradient = nothing; conv_op_node.gradient = nothing; loss_node_op.gradient = nothing;
        AD.forward!(graph_loss_op)
        AD.backward!(graph_loss_op; seed=1.0f0)
        @test X_node.gradient !== nothing && size(X_node.gradient) == size(X_val) && eltype(X_node.gradient) == Float32
        @test K_node.gradient !== nothing && size(K_node.gradient) == size(K_val) && eltype(K_node.gradient) == Float32

        # Test warstwy MF.Conv1D (z biasem i aktywacją relu)
        conv_layer = MF.Conv1D(
            (KW,),                         # kernel_size_tuple
            C_in => C_out;                 # channels_in_out
            activation=AD.relu,            # argument kluczowy
            use_bias=true                  # argument kluczowy (domyślnie true, ale można być jawnym)
        )
        # Ustaw wagi kernela i biasu na znane wartości (lub pozwól na losowe)
        # conv_layer.kernel.output .= K_val # Można przypisać, jeśli chcemy te same wagi co w teście operatora
        # if conv_layer.bias !== nothing; conv_layer.bias.output .= zeros(Float32, C_out); end
        
        # Forward pass dla warstwy
        # Warstwa Conv1D tworzy graf: X -> Conv1DOp -> (+) Bias -> Relu
        output_node_layer = conv_layer(X_node) # X_node jest wejściem (W, Cin, B)
        graph_layer = AD.topological_sort(output_node_layer)
        # Reset gradientów dla wszystkich parametrów warstwy
        for p in MF.layer_vars(conv_layer); p.gradient = nothing; end
        X_node.gradient = nothing # Również dla wejścia

        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (W_out_expected, C_out, B)
        @test eltype(output_node_layer.output) == Float32
        if C_out > 0 && W_out_expected > 0 && B > 0 # Tylko jeśli są elementy
             @test all(output_node_layer.output .>= 0.0f0) # Sprawdzenie efektu ReLU
        end

        # Test backward dla warstwy
        loss_node_layer = AD.sum(output_node_layer)
        graph_loss_layer = AD.topological_sort(loss_node_layer)
        # Reset gradientów
        for p in MF.layer_vars(conv_layer); p.gradient = nothing; end
        X_node.gradient = nothing; output_node_layer.gradient = nothing; loss_node_layer.gradient = nothing;
        
        AD.forward!(graph_loss_layer) # Potrzebne do ustawienia .output dla węzłów straty
        AD.backward!(graph_loss_layer; seed=1.0f0)

        @test X_node.gradient !== nothing && size(X_node.gradient) == size(X_val) && eltype(X_node.gradient) == Float32
        @test conv_layer.kernel.gradient !== nothing && size(conv_layer.kernel.gradient) == size(conv_layer.kernel.output) && eltype(conv_layer.kernel.gradient) == Float32
        if conv_layer.bias !== nothing
            @test conv_layer.bias.gradient !== nothing && size(conv_layer.bias.gradient) == size(conv_layer.bias.output) && eltype(conv_layer.bias.gradient) == Float32
        end
    end

    @testset "AD.MaxPool1DOp and MF.MaxPool1D Layer (with NNlib)" begin
        W_in, C, B = 20, 3, 2
        pool_size = (4,)
        
        X_val = randn(Float32, W_in, C, B)
        X_node = AD.Variable(X_val)

        # Test operatora
        pool_op_node = AD.MaxPool1DOp(X_node, pool_size)
        graph_op = AD.topological_sort(pool_op_node)
        AD.forward!(graph_op)
        
        W_out_expected = div(W_in - pool_size[1], pool_size[1]) + 1 # Stride = pool_size
        @test size(pool_op_node.output) == (W_out_expected, C, B)
        @test eltype(pool_op_node.output) == Float32

        # Test backward (kształt gradientu)
        loss_node_op = AD.sum(pool_op_node)
        graph_loss_op = AD.topological_sort(loss_node_op)
        X_node.gradient = nothing; pool_op_node.gradient = nothing; loss_node_op.gradient = nothing;
        AD.forward!(graph_loss_op)
        AD.backward!(graph_loss_op; seed=1.0f0)
        @test X_node.gradient !== nothing && size(X_node.gradient) == size(X_val) && eltype(X_node.gradient) == Float32

        # Test warstwy
        pool_layer = MF.MaxPool1D(pool_size)
        output_node_layer = pool_layer(X_node)
        graph_layer = AD.topological_sort(output_node_layer)
        X_node.gradient = nothing # Reset gradientu wejścia
        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (W_out_expected, C, B)
        @test eltype(output_node_layer.output) == Float32
    end

    @testset "AD.FlattenOp and MF.Flatten Layer" begin
        data_3d = randn(Float32, 10, 5, 2) # W, C, B
        X_node_3d = AD.Variable(data_3d)

        # Test operatora
        flatten_op_node = AD.FlattenOp(X_node_3d)
        graph_op = AD.topological_sort(flatten_op_node)
        AD.forward!(graph_op)
        expected_features = size(data_3d,1) * size(data_3d,2) # W*C
        expected_batch = size(data_3d,3) # B
        @test size(flatten_op_node.output) == (expected_features, expected_batch)
        @test eltype(flatten_op_node.output) == Float32

        # Test backward (kształt gradientu)
        loss_node_op = AD.sum(flatten_op_node) # Suma, aby uzyskać skalar
        graph_loss_op = AD.topological_sort(loss_node_op)
        X_node_3d.gradient = nothing; flatten_op_node.gradient = nothing; loss_node_op.gradient = nothing;
        AD.forward!(graph_loss_op)
        AD.backward!(graph_loss_op; seed=1.0f0)
        @test X_node_3d.gradient !== nothing && size(X_node_3d.gradient) == size(data_3d) && eltype(X_node_3d.gradient) == Float32

        # Test warstwy
        flatten_layer = MF.Flatten()
        output_node_layer = flatten_layer(X_node_3d)
        graph_layer = AD.topological_sort(output_node_layer)
        X_node_3d.gradient = nothing # Reset
        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (expected_features, expected_batch)
        @test eltype(output_node_layer.output) == Float32
    end

end