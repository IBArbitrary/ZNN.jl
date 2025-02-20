ACTIVATIONS = [
    :rrelu, :irelu, :abslu,
    :sigmoid_cart_comp, :softmax_polar_comp,
    :softmax_abs, :softmax_avg, :softmax_mult, :softmax_nested,
    :linear, :modrelu, :zrelu, :crelu, :cardioid
]

# Real valued
rrelu(z::Number) = real(z)
irelu(z::Number) = imag(z)
abslu(z::Number) = abs(z)
sigmoid_cart_comp(z::Number) = sigmoid(real(z) + imag(z))
softmax_polar_comp(z::Number) = (softmax(abs(z)) + softmax(angle(z))) / 2
softmax_abs(z::Number) = softmax(abs(z))
softmax_avg(z::Number) = softmax((real(z) + imag(z)) / 2)
softmax_mult(z::Number) = softmax(real(z)) * softmax(imag(z))
softmax_nested(z::Number) = softmax(softmax(real(z)) + softmax(imag(z)))

# Complex valued
linear(z::Number) = z
modrelu(z::Number, b::Float64=1.0) = relu(abs(z) + b) * z / abs(z)
zrelu(z::Number) = ifelse(0 <= angle(z) <= pi / 2, z, zero(z))
crelu(z::Number) = relu(real(z)) + 1im * relu(imag(z))
cardioid(z::Number) = (1 + cos(angle(z))) * z / 2