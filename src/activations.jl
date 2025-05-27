ACTIVATIONS = [
    :rlu, :ilu, :rrelu, :irelu, :abslu,
    :sigmoid_cart_comp, :softmax_polar_comp,
    :softmax_abs, :softmax_avg, :softmax_mult, :softmax_nested,
    :linear, :modrelu, :zrelu, :crelu, :cardioid,
    :cart_sigmoid, :cart_elu, :cart_exponential, :cart_hardsigmoid,
    :cart_relu, :cart_leakyrelu, :cart_selu, :cart_softplus,
    :cart_softsign, :cart_tanh, :cart_softmax,
    :pol_tanh, :pol_sigmoid, :pol_selu,
    :georgiou, :mvn, :complex_signum
]

# Real valued
rlu(z::Number) = real(z)
ilu(z::Number) = imag(z)
abslu(z::Number) = abs(z)
rrelu(z::Number) = relu(real(z))
irelu(z::Number) = relu(imag(z))
sigmoid_cart_comp(z::Number) = sigmoid(real(z) + imag(z))
softmax_polar_comp(z) = (softmax(abs.(z)) + softmax(angle.(z))) / 2
softmax_abs(z) = softmax(abs.(z))
softmax_avg(z) = softmax((real.(z) + imag.(z)) / 2)
softmax_mult(z) = softmax(real.(z)) * softmax(imag.(z))
softmax_nested(z) = softmax(softmax(real.(z)) + softmax(imag.(z)))

# Complex valued
linear(z::Number) = z

## ReLU based
modrelu(z::Number, b::AbstractFloat=1.0) = relu(abs(z) + b) * z / abs(z)
zrelu(z::Number) = ifelse(0 <= angle(z) <= pi / 2, z, zero(z))
crelu(z::Number) = relu(real(z)) + 1im * relu(imag(z))
cardioid(z::Number) = (1 + cos(angle(z))) * z / 2

## Cartesian form
cart_sigmoid(z::Number) = sigmoid(real(z)) + 1im * sigmoid(imag(z))
cart_elu(z::Number, α::AbstractFloat=1.0) = elu(real(z), α) + 1im * elu(imag(z), α)
cart_exponential(z::Number) = exp(real(z)) + 1im * exp(imag(z))
cart_hardsigmoid(z::Number) = hardsigmoid(real(z)) + 1im * hardsigmoid(imag(z))
cart_relu(z::Number) = relu(real(z)) + 1im * relu(imag(z))
cart_leakyrelu(z::Number, α::AbstractFloat=0.01) = leakyrelu(real(z), α) + 1im * leakyrelu(imag(z), α)
cart_selu(z::Number, α::AbstractFloat=1.0) = selu(real(z), α) + 1im * selu(imag(z), α)
cart_softplus(z::Number) = softplus(real(z)) + 1im * softplus(imag(z))
cart_softsign(z::Number) = softsign(real(z)) + 1im * softsign(imag(z))
cart_tanh(z::Number) = tanh(real(z)) + 1im * tanh(imag(z))
cart_softmax(z) = softmax(real.(z)) + 1im * softmax(imag.(z))

## Polar form
pol_tanh(z::Number) = tanh(abs(z)) * exp(1im * angle(z))
pol_sigmoid(z::Number) = sigmoid(abs(z)) * exp(1im * angle(z))
pol_selu(z::Number, α::AbstractFloat=1.0) = selu(abs(z), α) * exp(1im * angle(z))

## Phasor form
georgiou(z::Number, c::AbstractFloat=1.0, r::AbstractFloat=1.0) = z / (c + abs(z) / r)
mvn(z::Number, k::Int=3) = exp(2π * im * floor(Int, mod(angle(z), 2π) * k / (2π)) / k)
complex_signum(z::Number) = exp(1im * angle(z))
