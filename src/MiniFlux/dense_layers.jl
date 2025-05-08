function (layer::Dense)(x::GraphNode)
    println("Dense layer input shape: ", size(x.output))
    println("Dense layer weight shape: ", size(layer.W.output))
    println("Dense layer bias shape: ", size(layer.b.output))
    y = layer.W * x + layer.b
    return layer.activation(y)
end 