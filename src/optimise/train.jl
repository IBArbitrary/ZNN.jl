import Optimisers: _adjust, Leaf

gen_adjust!(tree, eta::Number) = foreach(st -> gen_adjust!(st, eta), tree)
gen_adjust!(tree; kw...) = foreach(st -> gen_adjust!(st; kw...), tree)

gen_adjust!(ℓ::Leaf, eta::Number) = (ℓ.rule = gen_adjust(ℓ.rule, eta);
nothing)
gen_adjust!(ℓ::Leaf; kw...) = (ℓ.rule = gen_adjust(ℓ.rule; kw...);
nothing)

gen_adjust(ℓ::Leaf, eta::Number) = Leaf(gen_adjust(ℓ.rule, eta), ℓ.state, ℓ.frozen)
gen_adjust(ℓ::Leaf; kw...) = Leaf(gen_adjust(ℓ.rule; kw...), ℓ.state, ℓ.frozen)

function gen_adjust(tree, eta::Number)
    t′ = fmap(copy, tree; exclude=maywrite)
    gen_adjust!(t′, eta)
    t′
end
function gen_adjust(tree; kw...)
    t′ = fmap(copy, tree; exclude=maywrite)
    gen_adjust!(t′; kw...)
    t′
end
gen_adjust(r::AbstractRule, eta::Number) = _adjust(r, (; eta))
gen_adjust(r::AbstractRule; kw...) = _adjust(r, NamedTuple(kw))

struct TrainingPhase
    epochs::Int
    rule::Optimisers.AbstractRule
    data_index::Int
    update_method::Function
    update_kwargs::Dict{Symbol,Any}

    TrainingPhase(
        epochs::Int, rule::Optimisers.AbstractRule,
        data_index::Int, update_method::Function,
        update_kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}()
    ) = new(epochs, rule, data_index, update_method, update_kwargs)

    TrainingPhase(
        epochs::Int, eta::Number,
        data_index::Int, update_method::Function,
        update_kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}()
    ) = new(epochs, GenDescent(eta), data_index, update_method, update_kwargs)
end
function Base.show(io::IO, tp::TrainingPhase)
    println(io, "TrainingPhase(")
    println(io, "  Epochs: $(tp.epochs)")
    println(io, "  Learning Rule(η): $(tp.rule)")
    println(io, "  Data Index: $(tp.data_index)")
    println(io, "  Update Method: $(nameof(tp.update_method))")
    if !isempty(tp.update_kwargs)
        println(io, "  Update Keyword Arguments: ")
        for (key, value) in tp.update_kwargs
            println(io, "    $(key): $(value)")
        end
    end
    println(io, ")")
end

struct TrainingPipeline
    phases::Vector{TrainingPhase}
    TrainingPipeline(phases::Vector{TrainingPhase}) = new(phases)
    TrainingPipeline(phases...) = new(collect(TrainingPhase, phases))
    TrainingPipeline() = new(Vector{TrainingPhase}())
end
function Base.show(io::IO, pipeline::TrainingPipeline)
    indent = "  "
    println(io, "TrainingPipeline(")
    for (i, phase) in enumerate(pipeline.phases)
        lines = split(sprint(show, phase), '\n')
        println(io, indent, "#$(i) TrainingPhase(")
        for line in lines[2:end-2]
            println(io, indent, line)
        end
        println(io, indent, lines[end-1], ",")
    end
    println(io, ")")
end

function add_phase!(pipeline::TrainingPipeline, phase::TrainingPhase)
    push!(pipeline.phases, phase)
    return pipeline
end

struct PhaseOutput
    loss::Vector{Float64}
    cbout::Dict{Symbol,Vector{Any}}
end

function run!(
    pipeline::TrainingPipeline, loss, model, data;
    before_step_cb=nothing, after_step_cb=nothing
)
    output_history = Vector{PhaseOutput}()
    for phase_idx in 1:length(pipeline.phases)
        phase_loss_history = Vector{Float64}()
        phase_cb_outputs = Dict{Symbol,Vector{Any}}(
            :beforecb => [],
            :aftercb => []
        )
        phase = pipeline.phases[phase_idx]
        for epoch in 1:phase.epochs
            current_data = data[phase.data_index]
            loss_value, callback_outputs = phase.update_method(
                loss,
                model,
                current_data,
                phase.rule,
                ; before_step_cb=before_step_cb,
                after_step_cb=after_step_cb,
                phase.update_kwargs...
            )
            push!(phase_loss_history, loss_value)
            if haskey(callback_outputs, "before_step_output")
                push!(phase_cb_outputs[:beforecb],
                    callback_outputs["before_step_output"])
            else
                push!(phase_cb_outputs[:beforecb], nothing)
            end
            if haskey(callback_outputs, "after_step_output")
                push!(phase_cb_outputs[:aftercb],
                    callback_outputs["after_step_output"])
            else
                push!(phase_cb_outputs[:aftercb], nothing)
            end
            println(
                "P: $(phase_idx); E: $epoch; L [$(phase.data_index)]: $loss_value"
            )
        end
        phase_output = PhaseOutput(phase_loss_history, phase_cb_outputs)
        push!(output_history, phase_output)
    end
    return output_history
end

function with_callbacks(train_step_fn)
    function wrapped_train_step!(
        loss, model, data, rule;
        before_step_cb=nothing, after_step_cb=nothing, kwargs...
    )
        callback_outputs = Dict{String,Any}()

        if before_step_cb !== nothing
            before_step_output = before_step_cb(
                ; loss=loss, model=model, data=data, rule=rule
            )
            if !isnothing(before_step_output)
                callback_outputs["before_step_output"] = before_step_output
            end
        end

        l = train_step_fn(loss, model, data, rule; kwargs...)

        if after_step_cb !== nothing
            after_step_output = after_step_cb(
                ; loss_value=l, loss=loss, model=model, data=data, rule=rule
            )
            if !isnothing(after_step_output)
                callback_outputs["after_step_output"] = after_step_output
            end
        end

        return l, callback_outputs
    end
    return wrapped_train_step!
end

function _train_euler!(loss, model, data, rule)
    (∇model, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    opt = Optimisers.setup(rule, model)
    opt, model = Optimisers.update!(opt, model, ∇model)
    return loss(model(data[1]), data[2])
end

function _train_norm_euler!(loss, model, data, rule; α=1.0)
    (∇model, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    gradvec, gradRe = destructure(∇model)
    ∇model_normed = gradRe(gradvec / (1 + α * norm(gradvec)))
    opt = Optimisers.setup(rule, model)
    opt, model = Optimisers.update!(opt, model, ∇model_normed)
    return loss(model(data[1]), data[2])
end

function _train_symplectic_euler!(
    loss, model, data, rule
)
    η = rule.eta
    opt = Optimisers.setup(rule, model)
    (∇1, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    gradvec1, gradRe1 = destructure(∇1)
    gradvec1 *= η
    gradvec1 -= 1im * imag.(gradvec1)
    ∇1 = gradRe1(gradvec1)
    gen_adjust!(opt, 1.0)
    opt, model = Optimisers.update!(opt, model, ∇1)
    (∇2, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    gradvec2, gradRe2 = destructure(∇2)
    gradvec2 *= η
    gradvec2 -= real.(gradvec2)
    ∇2 = gradRe2(gradvec2)
    gen_adjust!(opt, 1.0)
    opt, model = Optimisers.update!(opt, model, ∇2)
    return loss(model(data[1]), data[2])
end

function _train_rk4!(
    loss, model, data, rule
)
    η = rule.eta
    opt = Optimisers.setup(rule, model)
    model0, modelRe = destructure(model)
    (k1, _...) = Flux.gradient(model, data) do m, data
        loss(m(data[1]), data[2])
    end
    gen_adjust!(opt, η / 2)
    opt, model = Optimisers.update!(opt, model, k1)
    (k2, _...) = Flux.gradient(model, data) do m, data
        loss(m(data[1]), data[2])
    end
    model = modelRe(model0)
    gen_adjust!(opt, η / 2)
    opt, model = Optimisers.update!(opt, model, k2)
    (k3, _...) = Flux.gradient(model, data) do m, data
        loss(m(data[1]), data[2])
    end
    model = modelRe(model0)
    gen_adjust!(opt, η)
    opt, model = Optimisers.update!(opt, model, k3)
    (k4, _...) = Flux.gradient(model, data) do m, data
        loss(m(data[1]), data[2])
    end
    k1vec, kRe = destructure(k1)
    k2vec, _ = destructure(k2)
    k3vec, _ = destructure(k3)
    k4vec, _ = destructure(k4)
    kfvec = k1vec + 2 * k2vec + 2 * k3vec + k4vec
    kf = kRe(kfvec)
    model = modelRe(model0)
    gen_adjust!(opt, η / 6)
    opt, state = Optimisers.update!(opt, model, kf)
    return loss(model(data[1]), data[2])
end

train_euler! = with_callbacks(_train_euler!)
train_norm_euler! = with_callbacks(_train_norm_euler!)
train_symplectic_euler! = with_callbacks(_train_symplectic_euler!)
train_rk4! = with_callbacks(_train_rk4!)