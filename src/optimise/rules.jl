import Optimisers: AbstractRule, init, apply!

"""
    GenDescent(η = 1f-1)
    GenDescent(; [eta])

Classic gradient descent optimiser with learning rate `η` of any `Number` type.
For each parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
"""
struct GenDescent{T<:Number} <: AbstractRule
    eta::T
end

GenDescent(; eta = 1f-1) = GenDescent(eta)

init(o::GenDescent, x::AbstractArray) = nothing

function apply!(o::GenDescent, state, x, dx)
    η = ofeltype(x, o.eta)
    return state, @lazy dx * η
end

function Base.show(io::IO, o::GenDescent)
print(io, "GenDescent(")
show(io, o.eta)
print(io, ")")
end

