using Flux, Zygote, Optimisers, DifferentialEquations

struct IntegrationOutput{T, C}
    loss::T
    cbout::C
end

function model_derivative(u, model, loss, data, rule, t)
    _, re = destructure(model)
    model = re(u)
    (∇model_, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    η = rule.eta
    ∇model, _ = destructure(∇model_)
    return -η * ∇model
end

function loss_u(u, model, loss, data)
    _, re = destructure(model)
    model = re(u)
    return loss(model(data[1]), data[2])
end

function integrate!(pipeline::TrainingPipeline, loss, model, data;
    alg=Tsit5(), cb=icb_trajectory, kwargs...)
    output_history = Vector{IntegrationOutput}()
    t_cumulative = 0.0
    for phase_idx in 1:length(pipeline.phases)
        phase_losses = Vector{Float64}()
        phase_cb_outputs = Vector{Any}()
        phase = pipeline.phases[phase_idx]
        rule = phase.rule
        current_data = data[phase.data_index]
        tspan = (t_cumulative, t_cumulative + phase.epochs)
        t_cumulative += phase.epochs
        u0, re = destructure(model)
        function model_derivative_current(u, p, t)
            model, loss, data, rule = p
            return model_derivative(u, model, loss, data, rule, t)
        end
        loss_u_current(u) = loss_u(u, model, loss, current_data)
        prob = ODEProblem(model_derivative_current, u0, tspan,
            (model, loss, current_data, rule))
        sol = solve(prob, alg; kwargs...)
        current_loss = loss_u_current(sol.u[end])
        push!(phase_losses, current_loss)
        cb_output = cb(; solution=sol, loss=current_loss,
            model=re(sol.u[end]), data=current_data, rule=rule)
        push!(phase_cb_outputs, cb_output)
        push!(output_history, IntegrationOutput(phase_losses, phase_cb_outputs))
        println("P: $(phase_idx); L [$(phase.data_index)]: $current_loss; tspan: $tspan")
        model = re(sol.u[end])
    end
    return output_history
end

function my_callback(; loss, model, data, rule, u, t)
    println("Callback: Loss = $(loss), Eta = $(rule.eta)")
    return (; final_loss = loss,
        final_model_norm = norm(destructure(model)[1]), u=u, t=t)
end
