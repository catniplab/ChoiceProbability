"""
Experiments to compare the estimators
"""

using Revise
using ChoiceProbability
using Random, Distributions, LinearAlgebra
using JLD2, Dates
using ProgressMeter

nMC = 200;
cpr = 0.5:0.05:0.99;
dim = 20;

maps = [
    ((x, y) -> CP(x, y, :LR), "LR"),
    #((x, y) -> CP(x, y, LDA), "LDA"),
    ((x, y) -> CP(x, y, :ENLRCV), "L2 LR"),
    ((x, y) -> jackknifeCP(x, y, :LR)[1], "Jack-LR"),
    ((x, y) -> half_and_half_CP(x, y, :ENLRCV), "1/2 & 1/2 L2 LR"),
    #((x, y) -> pooledCP(x, y, ChoiceProbability.ENLRCV), "Pooled L2 LR (2-fold)"),
    ((x, y) -> pooledCP(x, y, :ENLRCV, nfolds=2), "Pooled L2 LR (2-fold)"),
    ((x, y) -> pooledCP(x, y, :ENLRCV, nfolds=5), "Pooled L2 LR (5-fold)"),
];

nr = [50, 100, 200, 400, 800, 1600, 3200];
cpea = Array{Any}(undef, (nMC, length(nr), length(cpr), length(maps)));

function mtNormalExperiment!(dim, maps, nMC, cpr, nr, cpea)
    for (kn, n) ∈ enumerate(nr)
        @showprogress for (ktcp, tcp) ∈ enumerate(cpr)
            mud = invCP_normal(tcp);
            ruvec = randn(dim);
            normalize!(ruvec);
            d1 = MvNormal(zeros(dim), I)
            d2 = MvNormal(mud * ruvec, I)
            x1 = rand(d1, n);
            x2 = rand(d2, n);
            Threads.@threads for k ∈ 1:nMC
                #rand!(d1, n, x1);
                #rand!(d2, n, x2);
                x1 = rand(d1, n);
                x2 = rand(d2, n);
                @inbounds for (kfh, (fh, name)) ∈ enumerate(maps)
                    cpea[k, kn, ktcp, kfh] = fh(x1, x2);
                end
            end
        end
    end
end

function fastBinaryPoissonRand(λ, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
	return (Random.randexp(rng) < λ) ? 1 : 0
end

function linearPop(z, C, nBins)
    b = -2
    λ = exp.(C * z' .+ b)
    y = zeros(UInt16, size(λ))
    for t = 1:nBins
        y += fastBinaryPoissonRand.(λ)
    end
    return y
end

function mtPoissonExperiment!(dim, maps, nMC, cpr, nr, cpea)
    nBins = 40;
    for (kn, n) ∈ enumerate(nr)
        @showprogress for (ktcp, tcp) ∈ enumerate(cpr)
            mud = invCP_normal(tcp);
            d1 = Normal(-1, 1)
            d2 = Normal(mud-1, 1)
            C = randn(dim) / 2
            Threads.@threads for k ∈ 1:nMC
                z1 = rand(d1, n);
                z2 = rand(d2, n);
                x1 = linearPop(z1, C, nBins);
                x2 = linearPop(z2, C, nBins);
                @inbounds for (kfh, (fh, name)) ∈ enumerate(maps)
                    cpea[k, kn, ktcp, kfh] = fh(x1, x2);
                end
            end
        end
    end
end

using Plots

function plotResults(dim, maps, nMC, cpr, nr, cpea)
    nMaps = length(maps)
    layout = @layout grid(2, nMaps)

    p = plot(
        size=(1800,600),
        titlefontsize=8,
        legend = false,
        legendfontsize=5,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        xlim = (0.4, 1),
        xticks = 0.5:0.1:1,
        yticks = 0.5:0.1:1,
        xlabel = "true CP",
        ylabel = "estimated CP",
        aspect_ratio = 1,
        layout = layout,
        #layout = length(nr) * length(maps),
        framestyle = :box
    )

    #for (kn, n) ∈ enumerate(nr)
        for (kfh, (fh, name)) ∈ enumerate(maps)
            #plot!(p[kn,kfh], title="$(dim)D $name n=$(n)")
            plot!(p[kfh], title="$(dim)D $name", guidefonthalign=:right)
            plot!(p[kfh], [0, 1], [0, 1], label=:none, linewidth=2, linecolor=:gray)
            if kfh > 1
                plot!(p[kfh], yformatter=_->"")
                plot!(p[kfh+nMaps], yformatter=_->"")
            end
            plot!(p[kfh+nMaps], widen=false)
        end
    #end

    #for (ktcp, tcp) ∈ enumerate(cpr)
        for (kfh, (fh, name)) ∈ enumerate(maps)
            for (kn, n) ∈ enumerate(nr)
                cpe = cpea[:, kn, :, kfh]
                cpe = cpea[:, kn, :, kfh]
                mcpe = vec(mean(cpe, dims=1))
                scpe = vec(std(cpe, dims=1))

                #scatter!(p[kfh], tcp * ones(length(cpe)), cpe, markeralpha=0.1, markercolor=:black, markerstrokewidth=0)
                #lot!(p[kfh], [tcp, tcp], [mcpe.-2scpe, mcpe.+2scpe], label="2SD", linewidth=1.5, linecolor=:orange)
                xa = cpr .+ 0.005 * (kn - length(nr)/2)
                for ktcp = 1:length(cpr)
                    plot!(p[kfh], [xa[ktcp], xa[ktcp]], [mcpe[ktcp].-2scpe[ktcp], mcpe[ktcp]], label=:none, color=:gray)
                    #plot!(p[kfh], [xa[ktcp], xa[ktcp]], [mcpe[ktcp].-2scpe[ktcp], mcpe[ktcp].+2scpe[ktcp]], label=:none, color=:gray)
                end
                plot!(p[kfh], xa, mcpe, marker=(:diamond, 3), label="n=$(n)")
            end
            plot!(p[kfh], ylim = (0.4, 1))
        end
    #end
        for (kfh, (fh, name)) ∈ enumerate(maps)
            for (kn, n) ∈ enumerate(nr)
                cpe = cpea[:, kn, :, kfh]
                cpe = cpea[:, kn, :, kfh]
                mcpe = vec(mean(cpe, dims=1))
                scpe = vec(std(cpe, dims=1))

                #scatter!(p[kfh], tcp * ones(length(cpe)), cpe, markeralpha=0.1, markercolor=:black, markerstrokewidth=0)
                #lot!(p[kfh], [tcp, tcp], [mcpe.-2scpe, mcpe.+2scpe], label="2SD", linewidth=1.5, linecolor=:orange)
                xa = cpr .+ 0.005 * (kn - length(nr)/2)
                for ktcp = 1:length(cpr)
                    plot!(p[kfh+nMaps], [xa[ktcp], xa[ktcp]], [mcpe[ktcp].-2scpe[ktcp], mcpe[ktcp]] .- cpr[ktcp], label=:none, color=:gray)
                    #plot!(p[kfh], [xa[ktcp], xa[ktcp]], [mcpe[ktcp].-2scpe[ktcp], mcpe[ktcp].+2scpe[ktcp]], label=:none, color=:gray)
                end
                plot!(p[kfh+nMaps], xa, mcpe - cpr, marker=(:diamond, 3), label="n=$(n)")
            end
            plot!(p[kfh+nMaps], yticks = -0.3:0.1:0.3)
            plot!(p[kfh+nMaps], ylim = (-0.3, 0.3))
        end
    plot!(p[1], legend = :bottomright)
    plot!(p[1+nMaps], legend = :bottomleft)
    ylabel!(p[2], "estimated CP");
    ylabel!(p[2+nMaps], "bias");
    xlabel!(p[1+nMaps], "true CP");
    display(p)
    return p
end

dateStr = Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS")
codeStr = split(split(@__FILE__, '/')[end], '.')[1]

@time nBins = mtPoissonExperiment!(dim, maps, nMC, cpr, nr, cpea)
jldsave("$(dateStr)_$(codeStr)_Poisson_output.jld2"; maps, nr, cpea, nMC, cpr, dim)

p = plotResults(dim, maps, nMC, cpr, nr, cpea)
savefig(p, "$(dateStr)_$(codeStr)_Poisson_output.pdf")

@time mtNormalExperiment!(dim, maps, nMC, cpr, nr, cpea)
jldsave("$(dateStr)_$(codeStr)_Normal_output.jld2"; maps, nr, cpea, nMC, cpr, dim)

p = plotResults(dim, maps, nMC, cpr, nr, cpea)
savefig(p, "$(dateStr)_$(codeStr)_Normal_output.pdf")

#jld = load("20220226_105113_exp1_Poisson_output.jld2")
#plotResults(dim, maps, nMC, cpr, nr, jld["cpea"])

"Check if the Poisson population is reasonable"
n = 1000
nBins = 40
p = Array{Any}(undef, length(cpr))
for (ktcp, tcp) ∈ enumerate(cpr)
#tcp = 0.7
    mud = invCP_normal(tcp);
    @show mud
    d1 = Normal(-1, 1)
    d2 = Normal(mud-1, 1)
    C = randn(dim) / 2
    z1 = rand(d1, n);
    z2 = rand(d2, n);
    x1 = linearPop(z1, C, nBins);
    x2 = linearPop(z2, C, nBins);
    println("$(tcp)")
    
    h1 = [count(x1 .== n) for n in 0:nBins]; h1 /= sum(h1);
    h2 = [count(x2 .== n) for n in 0:nBins]; h2 /= sum(h2);
    p[ktcp] = groupedbar(0:nBins, [h1 h2], bar_position = :dodge, bar_width=0.7, linewidth=0)
    xlabel!("spike count")
    ylabel!("probability")
    @show jackknifeCP(x1, x2, :LR)[1]
end
plot!(p..., size=(1000,800),
titlefontsize=8,
legend = false, layout = grid(5,2))