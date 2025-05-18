"""
    ComplexDense(in => out, σ=identity; bias=true, init=complex_glorot_uniform)
    ComplexDense(W::AbstractMatrix, [bias, σ])

Create a traditional fully connected layer with complex weights and biases,
whose forward pass is given by:

    y = σ.(W * x .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`

Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`complex_glorot_uniform`](@ref ZNN.complex_glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Examples
```jldoctest
julia> model = ComplexDense(5 => 2)
ComplexDense(5 => 2)  # 12 parameters

julia> model(rand32(5, 64)) |> size
(2, 64)

julia> model(rand32(5, 6, 4, 64)) |> size  # treated as three batch dimensions
(2, 6, 4, 64)

julia> model2 = ComplexDense(ones(2, 5), false, tanh)  # using provided weight matrix
ComplexDense(5 => 2, tanh; bias=false)  # 10 parameters

julia> model2(ones(5))
2-element Vector{ComplexF64}:
 0.9999092042625952 + 0.0im
 0.9999092042625952 + 0.0im

julia> Flux.trainables(model2)  # no trainable bias
1-element Vector{AbstractArray}:
 ComplexF32[1.0f0 + 0.0f0im 1.0f0 + 0.0f0im … 1.0f0 + 0.0f0im 1.0f0 + 0.0f0im; 1.0f0 + 0.0f0im 1.0f0 + 0.0f0im … 1.0f0 + 0.0f0im 1.0f0 + 0.0f0im]
```
"""
struct ComplexDense{F,M<:AbstractMatrix,B}
    weight::M
    bias::B
    σ::F

    function ComplexDense(W::AbstractMatrix, bias=true, σ=identity)
        W_complex = Complex{real(eltype(W))}.(W)
        b = if bias === true
            zeros(Complex{real(eltype(W))}, size(W, 1))
        elseif bias === false
            false
        else
            Complex{real(eltype(bias))}.(bias)
        end

        new{typeof(σ),typeof(W_complex),typeof(b)}(W_complex, b, σ)
    end
end

function ComplexDense((in, out)::Pair{<:Integer,<:Integer}, σ=identity;
    init=complex_glorot_uniform, bias=true)
    ComplexDense(init(out, in), bias, σ)
end

Flux.@layer ComplexDense

function (a::ComplexDense)(x::AbstractVecOrMat)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
    xT = if !isa(eltype(x), Complex)
        Complex{real(eltype(x))}.(x)
    else
        x
    end
    return NNlib.bias_act!(a.σ, a.weight * xT, a.bias)
end

function (a::ComplexDense)(x::AbstractArray)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::ComplexDense)
    print(io, "ComplexDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end