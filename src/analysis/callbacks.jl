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

# Integrate callbacks combiner function
function combine_integrate_callbacks(callback_list)
    function combined_callback(; solution, loss, model, data, rule)
        callback_outputs = []
        for cb in callback_list
            output = cb(; solution=solution, loss=loss,
                model=model, data=data, rule=rule
            )
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
    N_classes = size(y_test, 1)
    y_pred_class = Flux.onecold(y_pred, 1:N_classes)
    y_true_class = Flux.onecold(y_test, 1:N_classes)
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
        current_loss = loss(model(dataset[1]), dataset[2])
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
function cb_param_vector(; loss, model, data, rule, get_norm=false)
    param_vec, _ = destructure(model)
    if get_norm
        param_norm = norm(param_vec)
        return param_norm
    else
        return param_vec
    end
end


# Gradient Vector/Norm Callback
function cb_grad_vector(; loss, model, data, rule, get_norm=false)
    (grads, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    grad_vec, _ = destructure(grads)
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

# Solution trajectory
function icb_trajectory(; solution, loss, model, data, rule)
    return (u = solution.u, t = solution.t)
end

# Multi-dataset Loss Callback for integration - Returns list of losses
function icb_multidataset_loss(; solution, loss, model, data, rule, datasets)
    if isempty(datasets)
        @warn "No datasets provided for icb_multidataset_loss callback."
        return []
    end
    dataset_losses = []
    u = solution.u
    _, re = destructure(model)
    for d in datasets
        losses = []
        for vec ∈ u
            m = re(vec)
            push!(
                losses,
                loss(m(d[1]), d[2])
            )
        end
        push!(dataset_losses, losses)
    end
    return dataset_losses
end

# Parameter Vector/Norm Callback for integration
function icb_param_vector(; solution, loss, model, data, rule, get_norm=false)
    param_vecs = solution.u
    if get_norm
        param_norms = norm.(param_vecs)
        return param_norms
    else
        return param_vecs
    end
end


# Gradient Vector/Norm Callback for integration
function icb_grad_vector(; solution, loss, model, data, rule, get_norm=false)
    vecs = solution.u
    grad_out = []
    for u ∈ vecs
        (grads, _...) = Flux.gradient(model, data) do m, d
            loss(m(d[1]), d[2])
        end
        grad_vec, _ = destructure(grads)
        if get_norm
            push!(grad_out, norm(grad_vec))
        else
            push!(grad_out, grad_vec)
        end
    end
    return grad_out
end

# Model Accuracy Callback for integration
function icb_test_accuracy(; solution, loss, model, data, rule, test_data)
    if isnothing(test_data)
        @warn "No test_data provided for icb_test_accuracy callback."
        return
    end
    vecs = solution.u
    _, re = destructure(model)
    test_accuracy = []
    for u ∈ vecs
        model = re(u)
        push!(test_accuracy, accuracy(model, test_data))
    end
    return test_accuracy
end