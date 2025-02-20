# generates a Dense layer with same architecture as a ComplexDense layer
function complex_to_real_dense(
    complex_dense_layer::ComplexDense;
    init_method=:random
)
    W_complex = complex_dense_layer.weight
    b_complex = complex_dense_layer.bias
    σ = complex_dense_layer.σ
    in_size_complex = size(complex_dense_layer.weight, 2)
    out_size_complex = size(complex_dense_layer.weight, 1)
    if init_method == :random
        return Dense(in_size_complex => out_size_complex, σ, bias=(b_complex !== false))
    elseif init_method isa Function
        W_real_dense = init_method.(W_complex)
        b_real_dense = (b_complex !== false) ? init_method.(b_complex) : false
    else
        error("Invalid init_method: $init_method. Provide :random or a function for initialization.")
    end
    if init_method != :random
        init_W = (dims...) -> W_real_dense
        return Dense(in_size_complex => out_size_complex, σ;
            init=init_W,
            bias=b_real_dense)
    end
end

# converts a chain with ComplexDense layers to Dense layers
function complex_to_real_chain(
    complex_chain::Chain;
    init_method=:random
)
    real_layers = []
    for layer in complex_chain.layers
        if layer isa ComplexDense
            push!(real_layers, complex_to_real_dense(layer; init_method=init_method))
        elseif layer isa Chain
            push!(real_layers, convert_complex_chain_to_real_chain(layer; init_method=init_method))
        else
            push!(real_layers, layer)
        end
    end
    return Chain(real_layers...)
end

# converts a Dense layer to ComplexDense layer
function real_to_complex_dense(
    real_dense_layer::Dense;
    init_method=:zero_imag
)
    W_real = real_dense_layer.weight
    b_real = real_dense_layer.bias
    σ = real_dense_layer.σ
    in_size_real = size(real_dense_layer.weight, 2)
    out_size_real = size(real_dense_layer.weight, 1)
    local W_complex
    local b_complex
    if init_method == :zero_imag
        W_complex = W_real
        b_complex = b_real
    elseif init_method isa Function
        W_complex = init_method(out_size_real, in_size_real)
        b_complex = if b_real === false
            false
        else
            init_method(out_size_real)
        end
    else
        error("Invalid init_method: $init_method. Provide :zero_imag or a function for complex initialization.")
    end
    return ComplexDense(W_complex, b_complex, σ)
end

# converts a Chain with Dense layers to Chain with ComplexDense layers
function real_to_complex_chain(
    real_chain::Chain;
    init_method=:zero_imag
)
    complex_layers = []
    for layer in real_chain.layers
        if layer isa Dense
            push!(complex_layers, real_to_complex_dense(layer; init_method=init_method))
        elseif layer isa Chain
            push!(complex_layers, real_to_complex_chain(layer; init_method=init_method))
        else
            push!(complex_layers, layer)
        end
    end
    return Chain(complex_layers...)
end