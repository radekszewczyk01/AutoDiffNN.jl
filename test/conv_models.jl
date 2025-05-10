using Test
using myExample
using Random
using TestImages
using ImageTransformations  # For imresize

const AD = myExample.AutoDiff
const MF = myExample.MiniFlux

# Load a test image
img = Float64.(testimage("cameraman"))
img = reshape(img, size(img)..., 1, 1)  # Add channel and batch dimensions

@testset "MiniFlux CNN Test" begin
    # Create a simple CNN
    conv1 = MF.Conv2D(1, 16, (3,3), activation=AD.relu, padding=(1,1))
    pool1 = MF.MaxPool2D((2,2))
    conv2 = MF.Conv2D(16, 32, (3,3), activation=AD.relu, padding=(1,1))
    pool2 = MF.MaxPool2D((2,2))
    flatten = MF.Flatten()
    dense1 = MF.Dense(32 * 128 * 128, 10, AD.relu)
    dense2 = MF.Dense(10, 2, AD.linear)

    # Create model
    model = MF.Model(
        [conv1, pool1, conv2, pool2, flatten, dense1, dense2],
        [conv1.W, conv1.b, conv2.W, conv2.b, dense1.W, dense1.b, dense2.W, dense2.b]
    )

    # Test forward pass
    x = AD.Variable(img, name="x")
    y = model(x)
    output = AD.forward!(AD.topological_sort(y))
    @test size(output) == (2, 1)  # 2 classes, batch size 1

    # Test backward pass
    target = AD.Variable([1.0, 0.0], name="y")  # One-hot encoded target
    loss = MF.mse_loss(target, y)
    graph = AD.topological_sort(loss)
    AD.forward!(graph)
    AD.backward!(graph)

    # Check gradients
    @test !isnothing(conv1.W.gradient)
    @test !isnothing(conv1.b.gradient)
    @test !isnothing(conv2.W.gradient)
    @test !isnothing(conv2.b.gradient)
    @test !isnothing(dense1.W.gradient)
    @test !isnothing(dense1.b.gradient)
    @test !isnothing(dense2.W.gradient)
    @test !isnothing(dense2.b.gradient)
end

# Test with multiple images
@testset "MiniFlux CNN Batch Test" begin
    # Create a batch of images
    batch_size = 4
    batch = zeros(256, 256, 1, batch_size)
    
    for i in 1:batch_size
        # Get image and resize it to match batch dimensions
        img = Float64.(testimage("cameraman"))
        # Resize image to 256x256 using simple interpolation
        img = imresize(img, (256, 256))
        # Add channel dimension
        img = reshape(img, size(img)..., 1)

        batch[:,:,:,i] = img
    end

    # Create a simple CNN
    conv1 = MF.Conv2D(1, 16, (3,3), activation=AD.relu, padding=(1,1))
    pool1 = MF.MaxPool2D((2,2))
    flatten = MF.Flatten()
    dense1 = MF.Dense(16 * 128 * 128, 2, AD.linear)

    # Create model
    model = MF.Model(
        [conv1, pool1, flatten, dense1],
        [conv1.W, conv1.b, dense1.W, dense1.b]
    )

    # Test forward pass
    x = AD.Variable(batch, name="x")
    y = model(x)
    output = AD.forward!(AD.topological_sort(y))

    @test size(output) == (2, batch_size)

    targets = AD.Variable([1.0 0.0; 0.0 1.0; 1.0 0.0; 0.0 1.0]', name="y")
    loss = MF.mse_loss(targets, y)
    graph = AD.topological_sort(loss)
    AD.forward!(graph)
    AD.backward!(graph)

    # Check gradients
    @test !isnothing(conv1.W.gradient)
    @test !isnothing(conv1.b.gradient)
    @test !isnothing(dense1.W.gradient)
    @test !isnothing(dense1.b.gradient)
end 