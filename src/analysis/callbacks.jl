# Callback Combiner Function
function combine_callbacks(callback_list)
    function combined_callback(; loss, model, data, rule)
        callback_outputs = []
        for cb in callback_list
            output = cb(; loss=loss, model=model, data=data, rule=rule)
            push!(callback_outputs, output)
        end
        return callback_outputs
    end
    return combined_callback
end

# Accuracy function
function accuracy(model, tdata)
    x_test, y_test = tdata
    y_pred = model(x_test)
    N_classes = size(y_test[1], 1)
    y_pred_class = onecold(y_pred, 1:N_classes)
    y_true_class = onecold(y_test, 1:N_classes)
    return mean(y_pred_class .== y_true_class) * 100
end

# Multi-dataset Loss Callback - Returns list of losses
function cb_multidataset_loss(; loss, model, data, rule, datasets)
    if isempty(datasets)
        @warn "No datasets provided for cb_multidataset_loss callback."
        return []
    end
    dataset_losses = []
    for dataset in datasets
        current_loss = loss(model, dataset...)
        push!(dataset_losses, current_loss)
    end
    return dataset_losses
end

# Covariance Matrix Callback
function cb_covariance(; loss, model, data, rule, k::Int=10, retsamp::Bool=false)
    Random.seed!(42)
    xdata, ydata = data
    ind = sample(1:size(xdata)[2], k, replace=false)
    xsample, ysample = xdata[:, ind], ydata[:, ind]
    sampledata = (xsample, ysample)
    cm = zeros(ComplexF64, k, k)
    params_model = Flux.params(model)
    grads = [
        Flux.gradient(
            () -> loss(model(xsample[:, k_]), ysample[:, k_]),
            params_model
        ) for k_ ∈ 1:k
    ]
    grad_vecs = [reduce(vcat, [
        vec(grads[k_][p]) for p in params_model if grads[k_][p] !== nothing
    ]) for k_ ∈ 1:k]
    for i ∈ 1:k
        for j ∈ 1:k
            cm[i, j] += dot(
                grad_vecs[i], grad_vecs[j]
            ) / (
                norm(grad_vecs[i]) * norm(grad_vecs[j])
            )
        end
    end
    if retsamp
        return cm, sampledata
    else
        return cm
    end
end


# Parameter Vector/Norm Callback
function cb_param_vector(; loss, model, data, rule, get_norm=true)
    param_vec, re = Flux.destructure(model)
    if get_norm
        param_norm = norm(param_vec)
        return param_norm
    else
        return param_vec
    end
end


# Gradient Vector/Norm Callback
function cb_grad_vector(; loss, model, data, rule, get_norm=true)
    x, y = data
    params = Flux.params(model)
    grads = Zygote.gradient(() -> loss(model, x, y), params)
    grad_vec, _ = Flux.destructure(grads)
    if get_norm
        grad_norm = norm(grad_vec)
        return grad_norm
    else
        return grad_vec
    end
end

# Model Accuracy Callback
function cb_test_accuracy(; loss, model, data, rule, test_data)
    if isnothing(test_data)
        @warn "No test_data provided for cb_test_accuracy callback."
        return
    end
    test_accuracy_val = accuracy(model, test_data)
    return test_accuracy_val
end

# Plane projection
function cb_plane_projection(; loss, model, data, rule, basis)
    e1, e2 = basis
    e1 /= norm(e1)
    e2 /= norm(e2)
    if round(abs(dot(e1, e2))) != 0
        @warn "Basis vectors are not orthogonal."
        return
    end
    param_vec, _ = destructure(model)
    s, t = real(dot(param_vec, e1)), real(dot(param_vec, e2))
    return (s, t)
end