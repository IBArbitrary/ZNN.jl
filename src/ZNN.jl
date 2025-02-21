module ZNN

using Random,
    Optimisers,
    Flux,
    ChainRulesCore
using Optimisers: @lazy, destructure, ofeltype
using Flux: relu, sigmoid, softmax
using LinearAlgebra: norm, rank, svd
using Statistics: mean

include("layers/basic.jl")
export ComplexDense

include("layers/conversion.jl")
export complex_to_real_dense, complex_to_real_chain,
    real_to_complex_dense, real_to_complex_chain

include("utils.jl")
export complex_glorot_uniform,
    complex_aug_glorot_uniform,
    are_linearly_independent,
    gram_schmidt,
    principal_components,
    filter_classifier_data

include("activations.jl")
for f in ACTIVATIONS
    @eval export $(f)
end

include("optimise/rules.jl")
export GenDescent

include("optimise/train.jl")
export gen_adjust!,
    TrainingPhase,
    TrainingPipeline, add_phase!, run!,
    train_euler!,
    train_norm_euler!,
    train_symplectic_euler!,
    train_rk4!

include("optimise/integrate.jl")
export model_derivative,
    integrate!

include("analysis/callbacks.jl")
export accuracy,
    combine_callbacks
    cb_multidataset_loss,
    cb_covariance,
    cb_param_vector,
    cb_grad_vector,
    cb_test_accuracy,
    combine_integrate_callbacks,
    icb_trajectory

include("data_generators.jl")
export GenerateLineData,
    GenerateSquareDiagonalData,
    GenerateSquareQuadrantData,
    GenerateCubeOctantData

end
