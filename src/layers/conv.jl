# helper function

################################################################################

struct ComplexConv{N,M,F,A<:AbstractArray,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function ComplexConv(
    w::AbstractArray{T,N}, b=true, σ=identity;
    stride=1, pad=0, dilation=1, groups=1) where {T,N}

    @assert size(w, N) % groups == 0 "Output channel dimension must be divisible by groups."
    stride = Flux.expand(Val(N - 2), stride)
    dilation = Flux.expand(Val(N - 2), dilation)
    pad = Flux.calc_padding(
        ComplexConv, pad, size(w)[1:N-2], dilation, stride
    )
    bias = Flux.create_bias(w, b, size(w, N))
    return ComplexConv(σ, w, bias, stride, pad, dilation, groups)
end

function ComplexConv(
    k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ=identity;
    init=complex_glorot_uniform, stride=1, pad=0, dilation=1, groups=1,
    bias=true
) where {N}

    weight = Flux.convfilter(k, ch; init, groups)
    ComplexConv(weight, bias, σ; stride, pad, dilation, groups)
end

Flux.@layer ComplexConv

conv_dims(c::ComplexConv, x::AbstractArray) =
    DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation, groups=c.groups)

ChainRulesCore.@non_differentiable conv_dims(::Any, ::Any)

_channels_in(l::ComplexConv) = size(l.weight, ndims(l.weight) - 1) * l.groups
_channels_out(l::ComplexConv) = size(l.weight, ndims(l.weight))

function _conv_size_check(layer, x::AbstractArray)
    ndims(x) == ndims(layer.weight) || throw(DimensionMismatch(LazyString("layer ", layer,
      " expects ndims(input) == ", ndims(layer.weight), ", but got ", summary(x))))
    d = ndims(x)-1
    n = _channels_in(layer)
    size(x,d) == n || throw(DimensionMismatch(LazyString("layer ", layer,
      lazy" expects size(input, $d) == $n, but got ", summary(x))))
end

function (c::ComplexConv)(x::AbstractArray)
    _conv_size_check(c, x)
    cdims = conv_dims(c, x)
    xT = Flux._match_eltype(c, x)
    NNlib.bias_act!(
        c.σ, conv(xT, c.weight, cdims), Flux.conv_reshape_bias(c)
    )
end

function Base.show(io::IO, l::ComplexConv)
    print(io, "ComplexConv(", size(l.weight)[1:ndims(l.weight)-2])
    print(io, ", ", _channels_in(l), " => ", _channels_out(l))
    Flux._print_conv_opt(io, l)
    print(io, ")")
end

################################################################################

struct ComplexMeanPool{N,M}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function ComplexMeanPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
    stride = expand(Val(N), stride)
    pad = Flux.calc_padding(MeanPool, pad, k, 1, stride)
    return ComplexMeanPool(k, pad, stride)
end

function (m::ComplexMeanPool)(x)
    T = eltype(x)
    T_complex = T <: Real ? Complex{T} : T
    x_complex = T_complex.(x)
    Flux._pool_size_check(m, m.k, x_complex)
    pdims = NNlib.PoolDims(x_complex, m.k; padding=m.pad, stride=m.stride)
    return meanpool(x_complex, pdims)
end

function Base.show(io::IO, m::ComplexMeanPool)
    print(io, "ComplexMeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    print(io, ")")
end


################################################################################

struct ScalarMaxPool{N,M,F}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
    f::F # Scalar function f: Complex -> Real
end

function ScalarMaxPool(k::NTuple{N,Integer}, f::F; pad=0, stride=k) where {N,F}
    @assert isa(f, Function) "The argument `f` must be a function, but got $(typeof(f))"
    stride_ = Flux.expand(Val(N), stride)
    pad_ = Flux.calc_padding(ScalarMaxPool, pad, k, 1, stride_)
    return ScalarMaxPool(k, pad_, stride_, f)
end


function (m::ScalarMaxPool)(x)
    Flux._pool_size_check(m, m.k, x)
    pdims = NNlib.PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return scalarmaxpool(x, pdims; f=m.f)
end

function Base.show(io::IO, m::ScalarMaxPool)
    print(io, "ScalarMaxPool(", m.k, ", f=", m.f) # Added f to the print
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    print(io, ")")
end

################################################################################

struct ComplexScalarMaxPool{N,M,F}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
    f::F # Scalar function f: Complex -> Real
end

function ComplexScalarMaxPool(k::NTuple{N,Integer}, f::F; pad=0, stride=k) where {N,F}
    @assert isa(f, Function) "The argument `f` must be a function, but got $(typeof(f))"
    stride_ = Flux.expand(Val(N), stride)
    pad_ = Flux.calc_padding(ComplexScalarMaxPool, pad, k, 1, stride_)
    return ComplexScalarMaxPool(k, pad_, stride_, f)
end

function (m::ComplexScalarMaxPool)(x)
    T = eltype(x)
    T_complex = T <: Real ? Complex{T} : T
    x_complex = T_complex.(x)
    Flux._pool_size_check(m, m.k, x_complex)
    pdims = NNlib.PoolDims(x_complex, m.k; padding=m.pad, stride=m.stride)
    return scalarmaxpool(x_complex, pdims; f=m.f)
end

function Base.show(io::IO, m::ComplexScalarMaxPool)
    print(io, "ComplexScalarMaxPool(", m.k, ", f=", m.f) # Added f to the print
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    print(io, ")")
end

# #########################################################################
struct LpNormPool{N,M}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
    p::Real
end

function LpNormPool(k::NTuple{N,Integer}, p::Real=2; pad=0, stride=k) where {N}
    stride = Flux.expand(Val(N), stride)
    pad = Flux.calc_padding(LpNormPool, pad, k, 1, stride)
    return LpNormPool(k, pad, stride, p)
end

function (m::LpNormPool)(x)
    Flux._pool_size_check(m, m.k, x)
    pdims = NNlib.PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return NNlib.lpnormpool(x, pdims; p=m.p)
end

function Base.show(io::IO, m::LpNormPool)
    print(io, "LpNormPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", Flux._maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", Flux._maybetuple_string(m.stride))
    print(io, ", p=", m.p, ")")
end

###############################################################################3
struct ComplexMixedNormPool{N,M}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
    p::Real
    q::Real
end

function ComplexMixedNormPool(
    k::NTuple{N,Integer}, p::Real=2, q::Real=2; pad=0, stride=k
    ) where {N}
    stride = Flux.expand(Val(N), stride)
    pad = Flux.calc_padding(ComplexMixedNormPool, pad, k, 1, stride)
    return ComplexMixedNormPool(k, pad, stride, p, q)
end

function (m::ComplexMixedNormPool)(x)
    T = eltype(x)
    T_complex = T <: Real ? Complex{T} : T
    x_complex = T_complex.(x)
    Flux._pool_size_check(m, m.k, x_complex)
    pdims = NNlib.PoolDims(x_complex, m.k; padding=m.pad, stride=m.stride)
    return complexmixednormpool(x_complex, pdims; p=m.p, q=m.q)
end

function Base.show(io::IO, m::ComplexMixedNormPool)
    print(io, "ComplexMixedNormPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", Flux._maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", Flux._maybetuple_string(m.stride))
    print(io, ", p=", m.p, ")")
    print(io, ", q=", m.q, ")")
end
