## Pooling API
#
#  We provide the following generic methods, for 3d, 4d, and 5d tensors, calculating 1d,
#  2d and 3d pooling, based on the rank of the input tensors, in both mutating and
#  non-mutating auto-allocating variants:
#   - Pooling:
#     - scalarmaxpool(x, pdims)
#     - scalarmaxpool!(y, x, pdims)
#     - complexmixednormpool(x, pdims)
#     - complexmixednormpool!(y, x, pdims)
#   - Pooling input backprop
#     - ∇scalarmaxpool(dy, y, x, pdims)
#     - ∇scalarmaxpool!(dx, dy, y, x, pdims)
#     - ∇complexmixednormpool(dy, y, x, pdims)
#     - ∇complexmixednormpool!(dx, dy, y, x pdims)
#
#   All methods require a `PoolDims` object to define the dimensions and optional
#   elements of the convolution (stride, dilation, etc...), which is easily constructable
#   through something like `PoolDims(x, w)`.


# First, we will define mappings from the generic API names to our accelerated backend
# implementations.  At the moment this is only the direct implementation, however this
# exists here so that other packages (NNPACK, MAGMA, etc...) can override this easily.
for (front_name, backend) in (
	# This maps from public, front-facing name, to internal backend name
	:scalarmaxpool => :direct,
	:complexmixednormpool => :direct,
)

	# We only define 3d pooling primitives, we reshape lower down to get 1d and 2d pooling
	@eval begin
		function $(Symbol("$(front_name)!"))(
			y::AbstractArray{<:Any, 5}, x::AbstractArray{<:Any, 5},
			pdims::PoolDims; kwargs...)
			$(Symbol("$(front_name)_$(backend)!"))(y, x, pdims; kwargs...)
		end
	end
end

# Do the same for backprops
for (front_name, backend) in (
	:∇scalarmaxpool => :direct,
	:∇complexmixednormpool => :direct,
)
	@eval begin
		function $(Symbol("$(front_name)!"))(
			dx::AbstractArray{<:Any, 5}, dy::AbstractArray{<:Any, 5},
			y::AbstractArray{<:Any, 5}, x::AbstractArray{<:Any, 5},
			pdims::PoolDims; kwargs...)
			$(Symbol("$(front_name)_$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
		end
	end
end


# Our strategy for pooling is to reshape to an array with three spatial dimensions, which
# makes things MUCH EASIER for us on the backend side, and is in general pretty fast,
# since we can specialize on sizes.
for front_name in (:scalarmaxpool, :complexmixednormpool)
	for backend in (Symbol(), :_direct)
		for N in (3, 4)
			@eval begin
				function $(Symbol("$(front_name)$(backend)!"))(
					y::AbstractArray{<:Any, $N}, x::AbstractArray{<:Any, $N},
					pdims::PoolDims; kwargs...)
					$(Symbol("$(front_name)$(backend)!"))(
						insert_singleton_spatial_dimension(y, $(5 - N)),
						insert_singleton_spatial_dimension(x, $(5 - N)),
						insert_singleton_spatial_dimension(pdims, $(5 - N));
						kwargs...,
					)

					# We explicitly return `y` here, because the backend call
					# itself may return a reshaped view, which we don't want.
					return y
				end

				# backprops too
				function $(Symbol("∇$(front_name)$(backend)!"))(
					dx::AbstractArray{<:Any, $N}, dy::AbstractArray{<:Any, $N},
					y::AbstractArray{<:Any, $N}, x::AbstractArray{<:Any, $N},
					pdims::PoolDims; kwargs...)
					$(Symbol("∇$(front_name)$(backend)!"))(
						insert_singleton_spatial_dimension(dx, $(5 - N)),
						insert_singleton_spatial_dimension(dy, $(5 - N)),
						insert_singleton_spatial_dimension(y, $(5 - N)),
						insert_singleton_spatial_dimension(x, $(5 - N)),
						insert_singleton_spatial_dimension(pdims, $(5 - N));
						kwargs...,
					)

					# We explicitly return `dx` here, because the backend call
					# itself may return a reshaped view, which we don't want.
					return dx
				end
			end
		end
	end
end


# Finally, let's generate auto-allocating versions of all our functions, for all backends:
for backend in (Symbol(), :_direct)
	# First make auto-allocating versions of the basic pooling calls:
	for name in (:scalarmaxpool, :complexmixednormpool)
		@eval begin
			function $(Symbol("$(name)$(backend)"))(
				x::AbstractArray{<:Any, N},
				pdims::PoolDims; kwargs...) where {N}
				y = similar(x, output_size(pdims)..., channels_out(pdims), size(x, N))
				fill!(y, 0)
				return $(Symbol("$(name)$(backend)!"))(y, x, pdims; kwargs...)
			end

			# Backprops too
			function $(Symbol("∇$(name)$(backend)"))(
				dy::AbstractArray{<:Any, N}, y::AbstractArray{<:Any, N},
				x::AbstractArray{<:Any, N}, pdims::PoolDims;
				kwargs...) where {N}
				dx = similar(x, input_size(pdims)..., channels_in(pdims), size(dy, N))
				fill!(dx, 0)
				return $(Symbol("∇$(name)$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
			end
		end
	end
end

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)


"""
scalarmaxpool(x, k::NTuple{N, Integer}; pad=0, stride=k, f=abs)

Perform scalar max pool operation with window size `k` on input tensor `x` using
the scalar function `f`

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`
* `pad`: See [`pad_zeros`](@ref) for details.
* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.
* `f`: Scalar function f: C -> R. Default is `abs`
"""
function scalarmaxpool(
    x, k::NTuple{N, Integer}, f::Function=abs; pad=0, stride=k
    ) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return scalarmaxpool(x, pdims; f=f)
end


"""
complexmixednormpool(x, p::Real, q::Real, k::NTuple{N, Integer}; pad=0, stride=k)

Perform Lpq mixed pool operation with value of the Lpq norm `p` and `q`, and
window size `k` on input complex tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`
* `p` is restricted to `0 < p < Inf`.
* `q` is restricted to `0 < p < Inf`.
* `pad`: See [`pad_zeros`](@ref) for details.
* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

For all elements `x` in a size `k` window, complexmixednormpool computes `(∑ᵢ (Rxᵢ^q + Ixᵢ^q)^(1 / q))^(1 / p)` as an element of the output, where xᵢ = Rxᵢ + im*Ixᵢ

Thus `complexmixednormpool(x, 1, 1, k) ./ prod(k) ≈ meanpool(x, k)` and `mixednormpool(x, 2, 2, k).^2 ./ prod(k) ≈ meanpool(x.^2, k)`.
"""
function complexmixednormpool(x, p::Real, q::Real, k::NTuple{N, Integer}; pad = 0, stride = k) where {N}
	pow1 = p isa Integer ? p : convert(float(eltype(x)), p)
	(isinf(pow1) || pow1 < 0) && error("p value of Lpq norm pool expects `0 < p < Inf`, but p is $(pow1) now.")
    pow2 = q isa Integer ? q : convert(float(eltype(x)), q)
	(isinf(pow2) || pow2 < 0) && error("q value of Lpq norm pool expects `0 < q < Inf`, but q is $(pow2) now.")
	pdims = PoolDims(x, k; padding = expand(Val(N), pad), stride = expand(Val(N), stride))
	return complexmixednormpool(x, pdims; p = pow1, q = pow2)
end


for pool in [:scalarmaxpool, :complexmixednormpool]
	∇pool = Symbol(:∇, pool)
	pullback = Symbol(pool, :_pullback)
	@eval function ChainRulesCore.rrule(::typeof($pool), x, pdims::PoolDims; kw...)
		Ω = $pool(x, pdims; kw...)
		$pullback(Δ) = (NoTangent(), $∇pool(unthunk(Δ), Ω, x, pdims; kw...), NoTangent())
		return Ω, $pullback
	end
end
