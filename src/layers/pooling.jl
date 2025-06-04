using NNlib: insert_singleton_spatial_dimension,
    output_size, input_size, channels_out, channels_in,
    check_dims, calc_padding_regions, kernel_size,
    padding, dilation, stride


# Pooling is so similar, we abstract over meanpooling and maxpooling, simply replacing
# the inner loop operation and a few initialization parameters.
for name in (:scalarmax, :scalarlpnorm)
    @eval function $((Symbol("$(name)pool_direct!")))(
                    y::AbstractArray{<:Any, 5}, x::AbstractArray{<:Any, 5},
                    pdims::PoolDims; alpha=1, beta=0, kwargs...)
        $((Symbol("$(name)pool_direct!")))(
            y, x, pdims,
            Val(kernel_size(pdims)), Val(channels_out(pdims)),
            Val(padding(pdims)), Val(dilation(pdims)), Val(stride(pdims));
            alpha, beta, kwargs...)
        return y
    end

    @eval function $((Symbol("$(name)pool_direct!")))(
        y::AbstractArray{T,5}, x::AbstractArray{<:Any,5},
        pdims::PoolDims,
        # kernel size, channels out, padding, dilation, stride
        ::Val{K}, ::Val{C}, ::Val{P}, ::Val{D}, ::Val{S};
        alpha=1, beta=0, kwargs...
    ) where {T, K, C, P, D, S}
        @assert iszero(beta) "beta not supported yet"
        check_dims(size(x), size(y), pdims)

        width, height, depth = input_size(pdims)
        kernel_w, kernel_h, kernel_d = K
        pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = P
        dil_w, dil_h, dil_d = D
        stride_w, stride_h, stride_d = S

        # We use calc_padding_regions to split outselves up into separate regions that may or
        # may not need to worry about padding:
        padded_regions, central_region = calc_padding_regions(pdims)

        # A helper function to project from output (w, h) to input (input_w, input_h)
        @inline project(idx, stride, pad) = (idx - 1) * stride - pad + 1

        # If we're doing mean pooling, we represent division by kernel size by rolling it
        # into the `alpha` multiplier.
        # The type might change here, that's why we prepend the underscore
        # (does it make a difference, though?)
        _alpha = T(alpha)
        # _beta = T(beta)

        # A quick note on the array element types `T` and `R`:
        # Ideally, `T == R`, but in some edge-cases, this might not be the case
        # (e.g. with `ReverseDiff.TrackedArray`, see issue #484).
        # If the types differ, we will initialize variables (like `_alpha` above) with the
        # target eltype `T`.

        p = if $(name != :scalarlpnorm) 0 else
            !haskey(kwargs, :p) && error("scalarlpnormpool needs keyword argument `p`")
            kwargs[:p]
        end

        f = (haskey(kwargs, :f)) ? kwargs[:f] : abs

        # Each loop, we initialize `m` to something, set that here.
        farginit = if haskey(kwargs, :fargmin)
            kwargs[:fargmin]
        else
            # Default fargmin handling (you can customize this)
            if $(name == :scalarmax)
                if T <: AbstractFloat
                    nextfloat(typemin(T))
                elseif T <: Integer
                    typemin(T)
                elseif f == abs
                    T(0)
                else
                    error("No default `fargmin` for type $T")
                end
            elseif $(name == :scalarlpnorm)
                zero(T)
            else
                error("Unimplemented codegen path")
            end
        end
        m_init = convert(T, farginit)

        # Start with the central region
        w_region, h_region, d_region = central_region

        @inbounds for batch_idx in 1:size(x, 5), c in 1:C
            for d in d_region
            pd = project(d, stride_d, pad_d_lo)
            for h in h_region
            ph = project(h, stride_h, pad_h_lo)
            for w in w_region
            pw = project(w, stride_w, pad_w_lo)
            m = m_init

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = pd + (kd - 1) * dil_d
                input_kh = ph + (kh - 1) * dil_h
                input_kw = pw + (kw - 1) * dil_w

                # This conditional will be optimized away at compile time
                if $(name == :scalarmax)
                    xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                    if f(xv) > f(m)
                        m = xv
                    end
                elseif $(name == :scalarlpnorm)
                    # y = (∑ᵢ xᵢ^p)^(1 / p), here to calculate ∑ᵢ xᵢ^p
                    m += f(x[input_kw, input_kh, input_kd, c, batch_idx])^p
                else
                    error("Unimplemented codegen path")
                end
            end

            # for lpnormpool, y = (∑ᵢ xᵢ^p)^(1 / p)
            m = $(name == :scalarlpnorm) ? m^(T(1) / p) : m

            y[w, h, d, c, batch_idx] = _alpha * m # + _beta * y[w, h, d, c, batch_idx]
            end
            end
            end
        end

        # Next, the padded regions
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for batch_idx in 1:size(x, 5), c in 1:C
                for d in d_region
                pd = project(d, stride_d, pad_d_lo)
                for h in h_region
                ph = project(h, stride_h, pad_h_lo)
                for w in w_region
                pw = project(w, stride_w, pad_w_lo)
                m = m_init

                for kd in 1:kernel_d
                    input_kd = pd + (kd - 1) * dil_d
                    if input_kd <= 0 || input_kd > depth
                        # add here condition for handling options for paded value handling
                        continue
                    end

                    for kh in 1:kernel_h
                        input_kh = ph + (kh - 1) * dil_h
                        if input_kh <= 0 || input_kh > height
                            # add here condition for handling options for paded value handling
                            continue
                        end

                        for kw in 1:kernel_w
                            input_kw = pw + (kw - 1) * dil_w
                            if input_kw <= 0 || input_kw > width
                                # add here condition for handling options for paded value handling
                                continue
                            end

                            if $(name == :scalarmax)
                                xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                                if f(xv) > f(m)
                                    m = xv
                                end
                            elseif $(name == :scalarlpnorm)
                                m += f(x[input_kw, input_kh, input_kd, c, batch_idx])^p
                            else
                                error("Unimplemented codegen path")
                            end
                        end
                    end
                end
                $(name == :scalarlpnorm) && (m = m^(T(1) / p))
                y[w, h, d, c, batch_idx] = _alpha * m # + _beta * y[w, h, d, c, batch_idx]
                end
                end
                end
            end
        end

        return y
    end

    @eval function $((Symbol("∇$(name)pool_direct!")))(
                    dx::AbstractArray{<:Any,5}, dy::AbstractArray{<:Any,5},
                    y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                    pdims::PoolDims; kwargs...)
        $((Symbol("∇$(name)pool_direct!")))(
            dx, dy, y, x, pdims, Val(kernel_size(pdims)); kwargs...)
        return dx
    end

    # Same story for gradients, and although this is very similar to the forward pass,
    # it's unfortunately different enough that I think we need a separate function.  :(
    @eval function $((Symbol("∇$(name)pool_direct!")))(
                    dx::AbstractArray{T,5}, dy::AbstractArray{<:Any,5},
                    y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                    pdims::PoolDims, ::Val{K}; # == kernel_size(pdims)
                    alpha=1, beta=0, kwargs...) where {T, K}
        check_dims(size(x), size(dy), pdims)

        width, height, depth = input_size(pdims)
        kernel_w, kernel_h, kernel_d = K
        out_c = channels_out(pdims)
        pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = padding(pdims)
        dil_w, dil_h, dil_d = dilation(pdims)
        stride_w, stride_h, stride_d = stride(pdims)

        # Concerning array eltypes `DX, DY, X, Y`, we want handle them like above, i.e.,
        # initialize everything with the left-hand-side type (target type).
        # Of course, ideally the types are all the same anyways.

        # We use calc_padding_regions to split outselves up into separate regions that
        # may or may not need to worry about padding:
        padded_regions, central_region = calc_padding_regions(pdims)

        # A helper function to project from output (w, h) to input (input_w, input_h)
        @inline project(idx, stride, pad) = (idx - 1) * stride - pad + 1

        # If we're doing mean pooling, we represent division by kernel size by rolling
        # it into the `_alpha` multiplier.
        _alpha = T(alpha)

        p = if $(name != :scalarlpnorm) 0 else
            !haskey(kwargs, :p) && error("scalarlpnormpool needs keyword argument `p`")
            kwargs[:p]
        end

        f = (haskey(kwargs, :f)) ? kwargs[:f] : abs

        # Start with the central region
        w_region, h_region, d_region = central_region
        @inbounds for batch_idx in 1:size(x, 5), c in 1:out_c
            for d in d_region
            pd = project(d, stride_d, pad_d_lo)
            for h in h_region
            ph = project(h, stride_h, pad_h_lo)
            for w in w_region
            pw = project(w, stride_w, pad_w_lo)

            # Grab the output at this index for future use
            y_idx = y[w, h, d, c, batch_idx]
            dy_idx = dy[w, h, d, c, batch_idx]
            scalarmaxpool_already_chose = false

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = pd + (kd - 1) * dil_d
                input_kh = ph + (kh - 1) * dil_h
                input_kw = pw + (kw - 1) * dil_w

                # This conditional will be optimized away at compile time,
                # or my name isn't shengdan jingyu
                x_idxs = (input_kw, input_kh, input_kd, c, batch_idx)
                x_ = x[x_idxs...]
                if $(name == :scalarmax)
                    if scalarmaxpool_already_chose
                        break
                    end
                    # If it's equal; this is the one we chose. We only choose one per
                    # kernel window, all other elements of dx must be zero.
                    # Uncomment line below if using with non-precise output (e.g. by NNPACK)
                    # if abs(y_idx - x[x_idxs...]) < 1e-5 && !maxpool_already_chose
                    if f(y_idx) ≈ f(x_)
                        dx[x_idxs...] += dy_idx * _alpha #+ _beta * dx[x_idxs...]
                        scalarmaxpool_already_chose = true
                    # Maxpooling does not support `beta` right now.  :(
                    # else
                    #    dx[x_idxs...] = T(0) + beta*dx[x_idxs...]
                    end
                elseif $(name == :scalarlpnorm)
                    # y = (∑ᵢ xᵢ^p)^(1 / p), ∂y/∂xᵢ = xᵢ^(p-1) × y^(1-p)
                    fprime(x) = Zygote.gradient(f, x)[1]
                    grad = f(x_)^(p-1) * y_idx^(1-p) * fprime(x_)
                    dx[x_idxs...] += dy_idx * grad
                else
                    error("Unimplemented codegen path")
                end
            end
            end
            end
            end
        end

        # Next, the padded regions
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for batch_idx in 1:size(x, 5), c in 1:out_c
                for d in d_region
                pd = project(d, stride_d, pad_d_lo)
                for h in h_region
                ph = project(h, stride_h, pad_h_lo)
                for w in w_region
                pw = project(w, stride_w, pad_w_lo)

                # Grab the incoming gradient at this index for future use
                y_idx = y[w, h, d, c, batch_idx]
                dy_idx = dy[w, h, d, c, batch_idx]
                scalarmaxpool_already_chose = false

                # In these loops, we have to check that we're not reaching off the edge,
                # we do so by putting in a bunch of conditionals.  :/
                for kd in 1:kernel_d
                    input_kd = pd + (kd - 1) * dil_d
                    if input_kd <= 0 || input_kd > depth
                        continue
                    end

                    for kh in 1:kernel_h
                        input_kh = ph + (kh - 1) * dil_h
                        if input_kh <= 0 || input_kh > height
                            continue
                        end

                        for kw in 1:kernel_w
                            input_kw = pw + (kw - 1) * dil_w
                            if input_kw <= 0 || input_kw > width
                                continue
                            end

                            # Same as above
                            # x_idxs = (input_kw, input_kh, input_kd, c, batch_idx)
                            if $(name == :scalarmax)
                                if scalarmaxpool_already_chose
                                    break
                                end
                                # If it's equal; this is the one we chose. We only choose one per
                                # kernel window, all other elements of dx must be zero.
                                # Uncomment line below if using with non-precise output (e.g. by NNPACK)
                                # if abs(y_idx - x[x_idxs...]) < 1e-5 && !maxpool_already_chose
                                if f(y_idx) ≈ f(x_)
                                    dx[x_idxs...] += dy_idx * _alpha #+ _beta * dx[x_idxs...]
                                    scalarmaxpool_already_chose = true
                                # Maxpooling does not support `beta` right now.  :(
                                # else
                                #    dx[x_idxs...] = T(0) + beta*dx[x_idxs...]
                                end
                            elseif $(name == :scalarlpnorm)
                                # y = (∑ᵢ xᵢ^p)^(1 / p), ∂y/∂xᵢ = xᵢ^(p-1) × y^(1-p)
                                fprime(x) = Zygote.gradient(f, x)[1]
                                grad = f(x_)^(p-1) * y_idx^(1-p) * fprime(x_)
                                dx[x_idxs...] += dy_idx * grad
                            else
                                error("Unimplemented codegen path")
                            end
                        end
                    end
                end
            end
            end
            end
            end
        end

        return dx
    end
end

for (front_name, backend) in (
    # This maps from public, front-facing name, to internal backend name
    :scalarmaxpool  => :direct,
    :scalarlpnormpool => :direct,
    )

    @eval begin
        function $(Symbol("$(front_name)!"))(
                y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                pdims::PoolDims; kwargs...)
            $(Symbol("$(front_name)_$(backend)!"))(y, x, pdims; kwargs...)
        end
    end
end

# Do the same for backprops
for (front_name, backend) in (
    :∇scalarmaxpool  => :direct,
    :∇scalarlpnormpool => :direct,
    )
    @eval begin
        function $(Symbol("$(front_name)!"))(
                        dx::AbstractArray{<:Any,5}, dy::AbstractArray{<:Any,5},
                        y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                        pdims::PoolDims; kwargs...)
            $(Symbol("$(front_name)_$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
        end
    end
end


# Our strategy for pooling is to reshape to an array with three spatial dimensions, which
# makes things MUCH EASIER for us on the backend side, and is in general pretty fast,
# since we can specialize on sizes.
for front_name in (:scalarmaxpool, :scalarlpnormpool)
    for backend in (Symbol(), :_direct)
        for N in (3, 4)
            @eval begin
                function $(Symbol("$(front_name)$(backend)!"))(
                                y::AbstractArray{<:Any,$N}, x::AbstractArray{<:Any,$N},
                                pdims::PoolDims; kwargs...)
                    $(Symbol("$(front_name)$(backend)!"))(
                        insert_singleton_spatial_dimension(y, $(5 - N)),
                        insert_singleton_spatial_dimension(x, $(5 - N)),
                        insert_singleton_spatial_dimension(pdims, $(5 - N));
                        kwargs...
                    )

                    # We explicitly return `y` here, because the backend call
                    # itself may return a reshaped view, which we don't want.
                    return y
                end

                # backprops too
                function $(Symbol("∇$(front_name)$(backend)!"))(
                                dx::AbstractArray{<:Any,$N}, dy::AbstractArray{<:Any,$N},
                                y::AbstractArray{<:Any,$N}, x::AbstractArray{<:Any,$N},
                                pdims::PoolDims; kwargs...)
                    $(Symbol("∇$(front_name)$(backend)!"))(
                        insert_singleton_spatial_dimension(dx, $(5 - N)),
                        insert_singleton_spatial_dimension(dy, $(5 - N)),
                        insert_singleton_spatial_dimension(y, $(5 - N)),
                        insert_singleton_spatial_dimension(x, $(5 - N)),
                        insert_singleton_spatial_dimension(pdims, $(5 - N));
                        kwargs...
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
    for name in (:scalarmaxpool, :scalarlpnormpool)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                            x::AbstractArray{<:Any,N},
                            pdims::PoolDims; kwargs...) where {N}
                y = similar(x, output_size(pdims)..., channels_out(pdims), size(x, N))
                fill!(y, 0)
                return $(Symbol("$(name)$(backend)!"))(y, x, pdims; kwargs...)
            end

            # Backprops too
            function $(Symbol("∇$(name)$(backend)"))(
                            dy::AbstractArray{<:Any,N}, y::AbstractArray{<:Any,N},
                            x::AbstractArray{<:Any,N}, pdims::PoolDims;
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
function scalarmaxpool(x, k::NTuple{N, Integer}, f::Function=abs; pad=0, stride=k) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return scalarmaxpool(x, pdims; f=f)
end

"""
lpnormpool(x, p::Real, k::NTuple{N, Integer}; pad=0, stride=k)

Perform Lp pool operation with value of the Lp norm `p` and window size `k` on input tensor `x`, also known as LPPool in pytorch.
This pooling operator from [Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks](https://arxiv.org/abs/1311.1780).

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`
* `p` is restricted to `0 < p < Inf`.
* `pad`: See [`pad_zeros`](@ref) for details.
* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

For all elements `x` in a size `k` window, lpnormpool computes `(∑ᵢ xᵢ^p)^(1 / p)` as an element of the output.

Thus `lpnormpool(x, 1, k) ./ prod(k) ≈ meanpool(x, k)` and `lpnormpool(x, 2, k).^2 ./ prod(k) ≈ meanpool(x.^2, k)`.
"""
function scalarlpnormpool(
    x, p::Real, k::NTuple{N, Integer}, f::Function=abs; pad=0, stride=k
    ) where {N}
    pow = p isa Integer ? p : convert(float(eltype(x)), p)
    (isinf(pow) || pow < 0) && error("p value of Lp norm pool expects `0 < p < Inf`, but p is $(pow) now.")
    pdims = PoolDims(x, k; padding=expand(Val(N), pad), stride=expand(Val(N), stride))
    return scalarlpnormpool(x, pdims; p=pow, f=f)
end


for pool in [:scalarmaxpool, :scalarlpnormpool]
    ∇pool = Symbol(:∇, pool)
    pullback = Symbol(pool, :_pullback)
    @eval function rrule(::typeof($pool), x, pdims::PoolDims; kw...)
        Ω = $pool(x, pdims; kw...)
        $pullback(Δ) = (NoTangent(), $∇pool(unthunk(Δ), Ω, x, pdims; kw...), NoTangent())
        return Ω, $pullback
    end
end

