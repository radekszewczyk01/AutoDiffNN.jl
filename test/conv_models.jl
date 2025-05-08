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

    println("we got here 1")
    # Create model
    model = MF.Model(
        [conv1, pool1, conv2, pool2, flatten, dense1, dense2],
        [conv1.W, conv1.b, conv2.W, conv2.b, dense1.W, dense1.b, dense2.W, dense2.b]
    )

    println("we got here 2")
    # Test forward pass
    x = AD.Variable(img, name="x")
    println("Input shape: ", size(img))
    println("we got here 3")
    y = model(x)
    println("we got here 4")
    output = AD.forward!(AD.topological_sort(y))
    println("Output shape: ", size(output))
    println("we got here 5")

    @test size(output) == (2, 1)  # 2 classes, batch size 1

    # Test backward pass
    target = AD.Variable([1.0, 0.0], name="y")  # One-hot encoded target
    loss = MF.mse_loss(target, y)
    graph = AD.topological_sort(loss)
    println("we got here 6")
    AD.forward!(graph)
    println("we got here 7")    
    AD.backward!(graph)
    println("we got here 8")

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
    println("Batch shape: ", size(batch))
    
    for i in 1:batch_size
        # Get image and resize it to match batch dimensions
        img = Float64.(testimage("cameraman"))
        println("Original image shape: ", size(img))
        # Resize image to 256x256 using simple interpolation
        img = imresize(img, (256, 256))
        println("Resized image shape: ", size(img))
        # Add channel dimension
        img = reshape(img, size(img)..., 1)
        println("Final image shape: ", size(img))
        println("Target slice shape: ", size(batch[:,:,:,i]))
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
    println("Input variable shape: ", size(x.output))
    y = model(x)
    output = AD.forward!(AD.topological_sort(y))
    println("Output shape: ", size(output))

    @test size(output) == (2, batch_size)  # 2 classes, batch_size images

    # Test backward pass
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