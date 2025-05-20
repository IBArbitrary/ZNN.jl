using ZNN, Flux, Zygote, Optimisers

"""
    decouple(model) -> flat_ps, Recouple(re, iscomplex)

Decouples the parameters of a Flux model into a flattened vector of real numbers, or returns the original destructure output if the model is real.

This function takes a Flux model as input and returns two values:

1. `flat_ps`: A flattened vector containing the real and imaginary parts of the model's parameters (if complex), or the original flattened vector (if real).
2. `Recouple(re, iscomplex)`: A `Recouple` object, which is a callable struct that can be used to reconstruct the original model. `iscomplex` is a boolean indicating whether the original model was complex.

# Example (Complex Model):
```jldoctest
julia> model = Chain(ZNN.ComplexDense(10 => 5), ZNN.ComplexDense(5 => 2, ZNN.abslu));
julia> flat_params, recouple_func = decouple(model)
(Float32[0.59718806, 0.07128436, 0.15699963, 0.11974091, 0.250331, 0.05502281, 0.5139062, 0.16525012, 0.2177522, -0.04312502  …  0.76398486, -0.22840318, -0.8594159, -0.2718692, -0.24220636, 0.38780046, -0.15959483, -0.010475202, 0.0, 0.0], Recouple(Chain, ..., 67, iscomplex=true))
julia> reconstructed_model = recouple_func(flat_params)
Chain(
  ComplexDense(10 => 5),                # 55 parameters
  ComplexDense(5 => 2, abslu),          # 12 parameters
)                   # Total: 4 arrays, 67 parameters, 744 bytes.
true
```

# Example (Real Model):
```jldoctest
julia> model = Chain(Dense(10 => 5), Dense(5 => 2));
julia> flat_params, recouple_func = decouple(model)
(Float32[-0.43762666, -0.26887077, 0.46195325, -0.19606185, 0.3489221, -0.15470281, 0.35640138, 0.2771568, 0.5384779, 0.09215268  …  0.67135745, -0.35611394, -0.5852366, 0.6434416, -0.77624345, -0.8529054, 0.9225483, 0.5444854, 0.0, 0.0], Recouple(Chain, ..., 67, iscomplex=false))

julia> reconstructed_model = recouple_func(flat_params)
Chain(
  Dense(10 => 5),                       # 55 parameters
  Dense(5 => 2),                        # 12 parameters
)                   # Total: 4 arrays, 67 parameters, 476 bytes.
true
```
"""
function decouple(model)
    ps, re = Flux.destructure(model)
    iscomplex = eltype(ps) <: Complex
    if iscomplex
        flat_ps = vcat(real.(ps), imag.(ps))
    else
        flat_ps = ps
    end
    return flat_ps, Recouple(re, iscomplex)
end


"""
    Recouple(restructure, iscomplex)

A callable struct returned by [`decouple`](@ref).  `re(flat_vec)` will reconstruct the original model with
new parameters from vector `flat_vec`.

If the model is complex, `flat_vec` should contain the real parts followed by the imaginary parts of the parameters.
If the model is real, `flat_vec` is just the original flattened parameter vector.

# Example:
```jldoctest
julia> model = Chain(ZNN.ComplexDense(10 => 5), ZNN.ComplexDense(5 => 2, ZNN.abslu));
julia> flat_params, recouple_func = decouple(model)
(Float32[0.59718806, 0.07128436, 0.15699963, 0.11974091, 0.250331, 0.05502281, 0.5139062, 0.16525012, 0.2177522, -0.04312502  …  0.76398486, -0.22840318, -0.8594159, -0.2718692, -0.24220636, 0.38780046, -0.15959483, -0.010475202, 0.0, 0.0], Recouple(Chain, ..., 67, iscomplex=true))
julia> reconstructed_model = recouple_func(flat_params)
Chain(
  ComplexDense(10 => 5),                # 55 parameters
  ComplexDense(5 => 2, abslu),          # 12 parameters
)                   # Total: 4 arrays, 67 parameters, 744 bytes.
julia> trainables(reconstructed_model) == trainables(model)
true
```
"""
struct Recouple
    restructure::Any
    iscomplex::Bool # Add iscomplex field
end
function (r::Recouple)(flat_vec)
    if r.iscomplex
        len = length(flat_vec) ÷ 2
        combined_ps = complex.(flat_vec[1:len], flat_vec[len+1:end])
    else
        combined_ps = flat_vec  # No conversion needed for real model
    end
    return r.restructure(combined_ps)
end

function Base.show(io::IO, r::Recouple)
    T = typeof(r.restructure).parameters[1]
    print(io, "Recouple(", T.name.name, ", ..., ", length(r.restructure), ", iscomplex=", r.iscomplex, ")")
end