"""
    complex_glorot_uniform([rng], size...; gain = 1) -> Array
    complex_glorot_uniform([rng]; kw...) -> Function

Return an `Array{ComplexF32}` of the given `size` containing random complex
numbers whose real and imaginary parts are independently drawn from a uniform
distribution on the interval ``[-x, x]``, where `x = gain * sqrt(6 / (fan_in + fan_out))`.

This method initialises the real and imaginary parts independently using the
`glorot_uniform` method.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> complex_glorot_uniform(3, 4) |> summary
"3×4 Matrix{ComplexF32}"

julia> round.(extrema(real.(complex_glorot_uniform(10, 100))), digits=3)
(-0.234f0, 0.233f0)

julia> round.(extrema(imag.(complex_glorot_uniform(10, 100))), digits=3)
(-0.234f0, 0.233f0)

julia> round.(extrema(real.(complex_glorot_uniform(100, 10))), digits=3)
(-0.234f0, 0.233f0)

julia> round.(extrema(imag.(complex_glorot_uniform(100, 10))), digits=3)
(-0.234f0, 0.233f0)

julia> round.(extrema(real.(complex_glorot_uniform(100, 100))), digits=3)
(-0.173f0, 0.173f0)

julia> round.(extrema(imag.(complex_glorot_uniform(100, 100))), digits=3)
(-0.173f0, 0.173f0)

julia> Dense(3 => 2, tanh; init = complex_glorot_uniform(MersenneTwister(1)))
Dense(3 => 2, tanh)  # 8 parameters

julia> ans.bias
2-element Vector{ComplexF32}:
 0.0f0 + 0.0f0im
 0.0f0 + 0.0f0im

julia> ComplexDense(3 => 2, tanh; init = complex_glorot_uniform(MersenneTwister(1)))
ComplexDense(3 => 2, tanh)  # 8 parameters

julia> ans.bias
2-element Vector{ComplexF32}:
 0.0f0 + 0.0f0im
 0.0f0 + 0.0f0im
```
"""
function complex_glorot_uniform(
    rng::AbstractRNG, dims::Integer...; gain::Real=1
)
    return Flux.glorot_uniform(rng, dims...; gain) +
           1im .* Flux.glorot_uniform(rng, dims...; gain)
end
complex_glorot_uniform(dims::Integer...; kw...) =
    complex_glorot_uniform(Random.default_rng(), dims...; kw...)
complex_glorot_uniform(
    rng::AbstractRNG=default_rng(); init_kwargs...
) = (dims...; kwargs...) -> complex_glorot_uniform(
    rng, dims...; init_kwargs..., kwargs...
)
ChainRulesCore.@non_differentiable complex_glorot_uniform(::Any...)

"""
    complex_aug_glorot_uniform([rng], size...; gain = 1) -> Array
    complex_aug_glorot_uniform([rng]; kw...) -> Function

Return an `Array{ComplexF32}` of the given `size` containing random complex
numbers whose imaginary part is zero and the real part are independently drawn
from a uniform distribution on the interval ``[-x, x]``, where
`x = gain * sqrt(6 / (fan_in + fan_out))`.

This method initialises the real part using the `glorot_uniform` method.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> complex_aug_glorot_uniform(3, 4) |> summary
"3×4 Matrix{ComplexF32}"

julia> round.(extrema(real.(complex_aug_glorot_uniform(10, 100))), digits=3)
(-0.234f0, 0.233f0)

julia> round.(extrema(real.(complex_aug_glorot_uniform(100, 100))), digits=3)
(-0.173f0, 0.173f0)

julia> Dense(3 => 2, tanh; init = complex_aug_glorot_uniform(MersenneTwister(1)))
Dense(3 => 2, tanh)  # 8 parameters

julia> ans.weight
2×3 Matrix{ComplexF32}:
  0.86443+0.0im   0.657681+0.0im   0.96326+0.0im
 0.998903+0.0im  -0.387398+0.0im  0.351353+0.0im
```
"""
function complex_aug_glorot_uniform(
    rng::AbstractRNG, dims::Integer...; gain::Real=1
)
    return 0im .+ Flux.glorot_uniform(rng, dims...; gain)
end
complex_aug_glorot_uniform(dims::Integer...; kw...) =
    complex_aug_glorot_uniform(Random.default_rng(), dims...; kw...)
complex_aug_glorot_uniform(
    rng::AbstractRNG=default_rng(); init_kwargs...
) = (dims...; kwargs...) -> complex_aug_glorot_uniform(
    rng, dims...; init_kwargs..., kwargs...
)
ChainRulesCore.@non_differentiable complex_aug_glorot_uniform(::Any...)

"""
    are_linearly_independent(vectors; tolerance = 1e-8) -> Bool

Check for linear independence of a set of vectors.

Determines if a collection of vectors is linearly independent using numerical rank computation.
Linearly independent vectors cannot be expressed as linear combinations of each other.

# Arguments
- `vectors`: An `AbstractVector` of `AbstractVector`s (e.g., `Vector{Vector{Float64}}`). Vectors must have the same dimension.
- `tolerance`: Numerical tolerance for rank calculation (default: `1e-8`).

# Returns
- `Bool`: `true` if vectors are linearly independent, `false` otherwise.  Returns `true` for an empty set of vectors.

# Examples
```jldoctest; setup = :(using LinearAlgebra)
julia> v1 = [1.0, 2.0, 3.0]; v2 = [2.0, 4.0, 6.0]; v3 = [1.0, 0.0, 0.0];

julia> are_linearly_independent([v1, v2])
false

julia> are_linearly_independent([v1, v3])
true

julia> are_linearly_independent([])
true
```
"""
function are_linearly_independent(
    vectors::AbstractVector{<:AbstractVector};
    tolerance=1e-8
)
    num_vectors = length(vectors)
    if num_vectors == 0
        return true
    end
    for v in vectors
        if norm(v) < tolerance
            return false
        end
    end
    matrix_v = hcat(vectors...)
    r = rank(matrix_v; atol=tolerance)
    return r == num_vectors
end

"""
    gram_schmidt(vectors) -> Vector{<:AbstractVector}

Orthonormalize a set of vectors using Gram-Schmidt process.

Computes an orthonormal basis for the span of the input vectors.

# Arguments
- `vectors`: `Vector` of `AbstractVector`s (e.g., `Vector{Vector{Float64}}`). Vectors must have the same dimension.

# Returns
- `Vector{<:AbstractVector}`: Orthonormal basis vectors.

# Examples
```jldoctest; setup = :(using LinearAlgebra)
julia> v1 = [1.0, 1.0, 0.0]; v2 = [1.0, 2.0, 0.0];

julia> basis = gram_schmidt([v1, v2])
2-element Vector{Vector{Float64}}:
 [0.7071067811865475, 0.7071067811865475, 0.0]
 [-0.7071067811865475, 0.7071067811865475, 0.0]

julia> round(abs(dot(basis[1], basis[2])), digits=8)
0.0
```
"""
function gram_schmidt(vectors::Vector{<:AbstractVector})
    n = length(vectors)
    Q = Vector{typeof(vectors[1])}(undef, n)

    for i in 1:n
        q = copy(vectors[i])
        for j in 1:i-1
            proj = (dot(Q[j], vectors[i]) / dot(Q[j], Q[j])) * Q[j]
            q -= proj
        end
        Q[i] = q / norm(q)
    end

    return Q
end

"""
    principal_components(vectors::Vector{<:AbstractVector}, N::Int) -> (variances, pcs)

Perform Principal Component Analysis (PCA) and return principal components and variances.

Computes the top `N` principal components (eigenvectors) and their corresponding
variances (eigenvalues) for a given set of input vectors.

# Arguments
- `vectors`: A `Vector` of `AbstractVector`s representing data points.
- `N`: The number of principal components to return.

# Returns
- `variances`: Variances (eigenvalues) of the top `N` principal components as a `Vector`.
- `pcs`: Principal components (eigenvectors) as a `Vector` of `Vector`s (top `N` components).

# Examples
```jldoctest; setup = :(using LinearAlgebra)
julia> vecs = [[1.0, 2.0], [1.5, 2.5], [3.0, 4.0]];

julia> variances, pcs = principal_components(vecs, 2);

julia> variances
2-element Vector{Float64}:
 4.333333333333335
 3.959711965660156e-31

julia> pcs
2-element Vector{Vector{Float64}}:
 [0.7071067811865474, 0.7071067811865477]
 [-0.7071067811865477, 0.7071067811865474]
```
"""
function principal_components(vectors::Vector{<:AbstractVector}, N::Int)
    X = transpose((reduce(hcat, tuple(vectors...))))
    avg = mean(X, dims=1)
    center = X .- avg
    U, S, V = svd(center, full=false)
    pcs = V[:, 1:N]
    stds = S[1:N]
    return (stds .^ 2, [pcs[:, i] for i in 1:N])
end

"""
    filter_classifier_data(condition, x, y, yr)

Filters classifier data based on a condition applied to the response variable.

# Arguments

*   `condition`: A function that takes an element of `yr` and returns `true` if the corresponding data point should be kept, `false` otherwise.
*   `x`: The feature matrix (data points in columns).
*   `y`: The target variable (corresponding to columns of `x`).
*   `yr`: The response variable used for filtering.

# Returns

A tuple `(x_, y_)` containing the filtered feature matrix and target variable.
"""
function filter_classifier_data(condition::Function, x, y, yr)
    LEN = length(yr)
    indices = filter(i -> condition(yr[i]), 1:LEN)
    x_ = selectdim(x, ndims(x), indices)
    y_ = y[:, indices]
    return (x_, y_)
end

function jacobi(f, x)
    y, back = Zygote.pullback(f, x)
    back(1)[1], back(im)[1]
end
  
function wirtinger(f, x)
    du, dv = jacobi(f, x)
    (du' + im*dv')/2, (du + im*dv)/2
end