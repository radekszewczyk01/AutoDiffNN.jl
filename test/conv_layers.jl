using Test
using Random
using myExample
using NNlib

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

# Fukcja pomocnicza dla testów
function get_output_and_grads(op_node::AD.GraphNode, inputs_data::Tuple)

    if ndims(op_node.output) > 0 && length(op_node.output) > 1
        loss_sim_node = AD.sum(op_node)
    else
        loss_sim_node = op_node
    end

    graph = AD.topological_sort(loss_sim_node)

    AD.forward!(graph)
    output_val = op_node.output

    for n in graph; if n isa AD.Variable || n isa AD.Operator; n.gradient = nothing; end; end

    AD.backward!(graph; seed=1.0f0)
    
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

        x_indices_batch = rand(1:vocab_size, seq_len, batch_size)
        x_indices_single = rand(1:vocab_size, seq_len)

        W_node = AD.Variable(W_val)
        
        # Test z batchem
        x_idx_node_batch = AD.Constant(x_indices_batch)
        embed_op_node_batch = AD.EmbeddingOp(W_node, x_idx_node_batch)
        graph_batch = AD.topological_sort(embed_op_node_batch)
        AD.forward!(graph_batch)
        output_batch = embed_op_node_batch.output

        @test size(output_batch) == (embed_dim, seq_len, batch_size)
        @test eltype(output_batch) == Float32
        @test output_batch[:, 1, 1] ≈ W_val[:, x_indices_batch[1, 1]]

        loss_node_batch = AD.sum(embed_op_node_batch)
        graph_loss_batch = AD.topological_sort(loss_node_batch)

        W_node.gradient = nothing; embed_op_node_batch.gradient = nothing; loss_node_batch.gradient = nothing

        AD.forward!(graph_loss_batch)
        AD.backward!(graph_loss_batch; seed=1.0f0)
        
        @test W_node.gradient !== nothing
        @test size(W_node.gradient) == size(W_val)
        @test eltype(W_node.gradient) == Float32

        x_idx_node_single = AD.Constant(x_indices_single)
        embed_op_node_single = AD.EmbeddingOp(W_node, x_idx_node_single)
        graph_single = AD.topological_sort(embed_op_node_single)
        AD.forward!(graph_single)
        output_single = embed_op_node_single.output
        @test size(output_single) == (embed_dim, seq_len)
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
        x_input_node = AD.Variable(x_indices_val)
        
        output_node = layer(x_input_node)
        
        graph = AD.topological_sort(output_node)
        AD.forward!(graph)
        
        @test output_node.output isa Array{Float32, 3}
        @test size(output_node.output) == (embed_dim, seq_len, batch_size)
    end

    @testset "AD.PermuteDimsOp and MF.Permute Layer" begin
        data_3d = randn(Float32, 10, 5, 2) # W, C, B
        data_node = AD.Variable(data_3d)
        
        # Test operatora
        perm_op_node = AD.PermuteDimsOp(data_node, (2,1,3))
        graph_op = AD.topological_sort(perm_op_node)
        AD.forward!(graph_op)
        @test size(perm_op_node.output) == (5, 10, 2)
        @test eltype(perm_op_node.output) == Float32

        # Test warstwy
        perm_layer = MF.Permute((2,1,3))
        output_node_layer = perm_layer(data_node)
        graph_layer = AD.topological_sort(output_node_layer)

        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (5, 10, 2)
        @test eltype(output_node_layer.output) == Float32

        loss_node = AD.sum(output_node_layer)
        graph_loss = AD.topological_sort(loss_node)
        data_node.gradient = nothing; output_node_layer.gradient = nothing; loss_node.gradient = nothing;
        AD.forward!(graph_loss)
        AD.backward!(graph_loss; seed=1.0f0)
        @test data_node.gradient !== nothing
        @test size(data_node.gradient) == size(data_3d)
        @test eltype(data_node.gradient) == Float32
    end

    @testset "AD.Conv1DOp and MF.Conv1D Layer" begin
        W_in, C_in, B = 10, 3, 2  # Szerokość, Kanały wej, Batch
        KW, C_out = 3, 5          # Szerokość Kernela, Kanały wyj
        
        X_val = randn(Float32, W_in, C_in, B)
        K_val = randn(Float32, KW, C_in, C_out)
        
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
            (KW,),
            C_in => C_out;
            activation=AD.relu,
            use_bias=true
        )

        output_node_layer = conv_layer(X_node)
        graph_layer = AD.topological_sort(output_node_layer)

        for p in MF.layer_vars(conv_layer); p.gradient = nothing; end
        X_node.gradient = nothing

        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (W_out_expected, C_out, B)
        @test eltype(output_node_layer.output) == Float32
        if C_out > 0 && W_out_expected > 0 && B > 0
             @test all(output_node_layer.output .>= 0.0f0)
        end

        loss_node_layer = AD.sum(output_node_layer)
        graph_loss_layer = AD.topological_sort(loss_node_layer)

        for p in MF.layer_vars(conv_layer); p.gradient = nothing; end
        X_node.gradient = nothing; output_node_layer.gradient = nothing; loss_node_layer.gradient = nothing;
        
        AD.forward!(graph_loss_layer)
        AD.backward!(graph_loss_layer; seed=1.0f0)

        @test X_node.gradient !== nothing && size(X_node.gradient) == size(X_val) && eltype(X_node.gradient) == Float32
        @test conv_layer.kernel.gradient !== nothing && size(conv_layer.kernel.gradient) == size(conv_layer.kernel.output) && eltype(conv_layer.kernel.gradient) == Float32
        if conv_layer.bias !== nothing
            @test conv_layer.bias.gradient !== nothing && size(conv_layer.bias.gradient) == size(conv_layer.bias.output) && eltype(conv_layer.bias.gradient) == Float32
        end
    end

    @testset "AD.MaxPool1DOp and MF.MaxPool1D Layer" begin
        W_in, C, B = 20, 3, 2
        pool_size = (4,)
        
        X_val = randn(Float32, W_in, C, B)
        X_node = AD.Variable(X_val)

        # Test operatora
        pool_op_node = AD.MaxPool1DOp(X_node, pool_size)
        graph_op = AD.topological_sort(pool_op_node)
        AD.forward!(graph_op)
        
        W_out_expected = div(W_in - pool_size[1], pool_size[1]) + 1
        @test size(pool_op_node.output) == (W_out_expected, C, B)
        @test eltype(pool_op_node.output) == Float32

        loss_node_op = AD.sum(pool_op_node)
        graph_loss_op = AD.topological_sort(loss_node_op)
        X_node.gradient = nothing; pool_op_node.gradient = nothing; loss_node_op.gradient = nothing;
        AD.forward!(graph_loss_op)
        AD.backward!(graph_loss_op; seed=1.0f0)
        @test X_node.gradient !== nothing && size(X_node.gradient) == size(X_val) && eltype(X_node.gradient) == Float32


        pool_layer = MF.MaxPool1D(pool_size)
        output_node_layer = pool_layer(X_node)
        graph_layer = AD.topological_sort(output_node_layer)
        X_node.gradient = nothing
        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (W_out_expected, C, B)
        @test eltype(output_node_layer.output) == Float32
    end

    @testset "AD.FlattenOp and MF.Flatten Layer" begin
        data_3d = randn(Float32, 10, 5, 2)
        X_node_3d = AD.Variable(data_3d)

        # Test operatora
        flatten_op_node = AD.FlattenOp(X_node_3d)
        graph_op = AD.topological_sort(flatten_op_node)
        AD.forward!(graph_op)
        expected_features = size(data_3d,1) * size(data_3d,2)
        expected_batch = size(data_3d,3) # B
        @test size(flatten_op_node.output) == (expected_features, expected_batch)
        @test eltype(flatten_op_node.output) == Float32

        # Test backward (kształt gradientu)
        loss_node_op = AD.sum(flatten_op_node)
        graph_loss_op = AD.topological_sort(loss_node_op)
        X_node_3d.gradient = nothing; flatten_op_node.gradient = nothing; loss_node_op.gradient = nothing;
        AD.forward!(graph_loss_op)
        AD.backward!(graph_loss_op; seed=1.0f0)
        @test X_node_3d.gradient !== nothing && size(X_node_3d.gradient) == size(data_3d) && eltype(X_node_3d.gradient) == Float32

        # Test warstwy
        flatten_layer = MF.Flatten()
        output_node_layer = flatten_layer(X_node_3d)
        graph_layer = AD.topological_sort(output_node_layer)
        X_node_3d.gradient = nothing
        AD.forward!(graph_layer)
        @test size(output_node_layer.output) == (expected_features, expected_batch)
        @test eltype(output_node_layer.output) == Float32
    end

end