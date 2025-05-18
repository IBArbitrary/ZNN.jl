struct IntegrationOutput{T, C}
    loss::T
    cbout::C
end

struct IntegrationHistory{L, C}
    loss::Vector{L}
    cbout::Vector{C}
end

function model_derivative(u, model, loss, data, rule, t)
    _, re = destructure(model)
    model = re(u)
    (∇model_, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    η = rule.eta / abs(rule.eta)
    ∇model, _ = destructure(∇model_)
    return -η * ∇model |> f64
end

function model_derivative_decoupled(u, model, loss, data, rule, t)
    _, re = decouple(model)
    model = re(u)
    (∇model_, _...) = Flux.gradient(model, data) do m, d
        loss(m(d[1]), d[2])
    end
    η = rule.eta / abs(rule.eta)
    ∇model_c, re2 = destructure(∇model_)
    model = re2(-η * ∇model_c)
    ∇model, _ = decouple(model)
    return ∇model |> f64
end

function loss_u(u, model, loss, data)
    _, re = destructure(model)
    model = re(u)
    return loss(model(data[1]), data[2])
end

function print_callback_(phase_idx, data_index, integrator, loss_value)
    println("P: $(phase_idx); T: $(integrator.t); L [$(data_index)]: $(loss_value(integrator.u))")
end

function integrate!(
    pipeline::TrainingPipeline, loss, model, data;
    alg=Tsit5(), rtol=1e-5, atol=1e-5, adaptive=false,
    icb=ZNN.icb_trajectory, dtmax=0.01, decouple=false, cb=nothing,
)
    phase_outputs = Vector{IntegrationOutput}()
    t_cumulative = 0.0
    for phase_idx in 1:length(pipeline.phases)
        phase_losses = Vector{Float64}()
        phase = pipeline.phases[phase_idx]
        rule = phase.rule
        current_data = data[phase.data_index]
        tspan = (t_cumulative, t_cumulative + phase.epochs * abs(rule.eta))
        t_cumulative += tspan[2]
        u0, re = destructure(model)
        function model_derivative_current(u, p, t)
            if decouple
                return model_derivative_decoupled(
                    u, model, loss, current_data, rule, t
                )
            else
                return model_derivative(
                    u, model, loss, current_data, rule ,t
                )
            end
        end
        loss_u_current(u) = loss_u(u, model, loss, current_data)
        cb_ = DiscreteCallback(
            (u, t, integrator) -> true,
            (integrator) -> print_callback_(
                phase_idx, phase.data_index, integrator, loss_u_current
                )
        )
        prob = ODEProblem(model_derivative_current, u0, tspan)
        if !isnothing(cb)
            cbs = CallbackSet(cb..., cb_)
        else
            cbs = CallbackSet(cb_,)
        end
        if !adaptive
            sol = solve(
                prob, alg, reltol=rtol, abstol=atol,
                dt=abs(rule.eta), adaptive=adaptive,
                dtmax=dtmax, callback=cbs, save_everystep=false
            )
        else
            sol = solve(
                prob, alg, reltol=rtol, abstol=atol,
                adaptive=adaptive, dtmax=dtmax, callback=cbs,
                save_everystep=false
            )
        end
        for u_ in sol.u
            push!(phase_losses, loss_u_current(u_))
        end
        cb_output = icb(; solution=sol, loss=loss,
            model=re(sol.u[end]), data=current_data, rule=rule)
        push!(phase_outputs, IntegrationOutput(phase_losses, cb_output))
        current_loss = loss_u_current(sol.u[end])
        # println("P: $(phase_idx); L [$(phase.data_index)]: $current_loss; tspan: $tspan")
        model = re(sol.u[end])
    end
    all_losses = [p.loss for p in phase_outputs]
    all_cbouts = [p.cbout for p in phase_outputs]
    return IntegrationHistory(all_losses, all_cbouts)
end