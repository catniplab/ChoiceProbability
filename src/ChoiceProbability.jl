"
Choice probability is used to quantify correlated variability
between neurons and a binary behavioral outcome measurement.
"
module ChoiceProbability

using Random, Distributions
using HypothesisTests: tiedrank_adj
using LinearAlgebra, GLM, GLMNet, MLBase
using QuadGK: quadgk

export CP
export invCP_normal
export half_and_half_CP, pooledCP, jackknifeCP
export invCP_normal, invCP_search

"Traditional CP estimator for 1D case"
function CP(x::AbstractVector, y::AbstractVector)
    (ranks, tieadj) = tiedrank_adj([x; y])
    nx = length(x)
    ny = length(y)
    U = sum(@view ranks[1:nx]) - nx*(nx+1)/2
    U = 1 - U / nx / ny
end

"Linear Population Choice Probability"
function CP(x::AbstractMatrix, y::AbstractMatrix, w::AbstractVector)
    return CP((w' * x)', (w' * y)')
end

"1D gaussian case has a closed form"
function CP(mu1::AbstractFloat, sigma1::AbstractFloat, mu2::AbstractFloat, sigma2::AbstractFloat)
    t = (mu1 - mu2) / (sigma1^2 + sigma2^2)^0.5
    return cdf(Normal(t, 1), 0)
end
CP(d1::Normal, d2::Normal) = CP(d1.μ, d1.σ, d2.μ, d2.σ)

function integrationRange(d1::Distribution, d2::Distribution, α::AbstractFloat=0.0001)
    @assert α >= 0
    @assert α <= 1
    thr = [quantile(d1, α), quantile(d1, 1-α), quantile(d2, α), quantile(d2, 1-α)];
    return (min(thr...), max(thr...))
end

"If both are discrete distributions, CP can be evaluated with a summation"
function CP(d1::DiscreteUnivariateDistribution, d2::DiscreteUnivariateDistribution, α::AbstractFloat=0.0001)
    thr = integrationRange(d1, d2, α)
    thr = thr[1]:thr[2]
    return sum((cdf.(d1, thr .- 1) + pdf.(d1, thr)/2) .* pdf.(d2, thr))
end

"General continuous distribution case, use QuadGK to numerically integrate"
function CP(d1::Distribution, d2::ContinuousUnivariateDistribution, α::AbstractFloat=0.0001)
    thr = integrationRange(d1, d2, α)
    return quadgk(x -> cdf.(d1, x) .* pdf.(d2, x), thr[1], thr[end])[1]
end

CP(d1::ContinuousUnivariateDistribution, d2::DiscreteUnivariateDistribution, α::AbstractFloat=0.0001) = 1 - CP(d2, d1, α)

"Returns mean difference of standard Normal distributions to achieve desired CP"
invCP_normal = cp -> invlogcdf(Normal(), log(cp)) * sqrt(2)

struct NotFoundError <: Exception end

"""Find the parameter that yields target CP via linear interpolation search
invCP(targetCP, baseDistribution, parameter -> targetDistribution, parameter range left, right)
it assumes that CP increases as parameter increases in value.

Use invCP_normal if you have two normal distributions.
"""
function invCP_search(targetCP::T, d1::Distribution, d2fun::Function, θ_0::Real, θ_1::Real, tol::Real=1e-3, maxEval::Number=50, cp_0=nothing, cp_1=nothing)::T where T<:Real
    if maxEval < 1
        throw(NotFoundError((θ_0, θ_1, cp_0, cp_1), "Ran out of evaluation bound"))
    end

    if isnothing(cp_0)
        cp_0 = CP(d1, d2fun(θ_0))
    end

    if isnothing(cp_1)
        cp_1 = CP(d1, d2fun(θ_1))
    end

    if cp_0 > targetCP
        throw(DomainError(cp_0, "CP at left boundary must be less than targetCP"))
    elseif cp_1 < targetCP
        throw(DomainError(cp_1, "CP at right boundary must be greater than targetCP"))
    end

    θ_new = (targetCP - cp_0) * (θ_1 - θ_0) / (cp_1 - cp_0)
    cp_new = CP(d1, d2fun(θ_new))

    if abs(cp_new - targetCP) <= tol
        return θ_new
    end

    if cp_new < targetCP
        return invCP_search(targetCP, d1, d2fun, θ_new, θ_1, tol, maxEval-1, cp_new, cp_1)
    else
        return invCP_search(targetCP, d1, d2fun, θ_0, θ_new, tol, maxEval-1, cp_0, cp_new)
    end
end

"""Definition of Linear Population Choice Probability given the linear map
Try a function that does regression, or internally defined :LDA, :LR, :ENLRCV for fun.
"""
function CP(x::AbstractMatrix, y::AbstractMatrix, findLinaerMap::Union{Function,Symbol})
    return CP(x, y, execFindLinearMap(findLinaerMap, x, y))
end

@inline function execFindLinearMap(findLinaerMap::Symbol, x::AbstractMatrix, y::AbstractMatrix)
    return eval(Expr(:call, findLinaerMap, x, y))
end

@inline function execFindLinearMap(findLinaerMap::Function, x::AbstractMatrix, y::AbstractMatrix)
    return findLinaerMap(x, y)
end

"Linear discriminant analysis"
function LDA(x::AbstractMatrix, y::AbstractMatrix)
    C1 = cov(x')
    C2 = cov(y')
    mu1 = mean(x, dims=2)
    mu2 = mean(y, dims=2)
    w = vec((C1 + C2) \ (mu2 - mu1))
    return w
end

"Plain logistic regression using GLM"
function LR(x::AbstractMatrix, y::AbstractMatrix)
    X = copy(hcat(x, y)');
    Y = vcat(zeros(size(x,2)), ones(size(y,2)));
    try
        m = glm(X, Y, Binomial(), LogitLink());
        return GLM.coef(m)
    catch e
        # Convert Errors to Warnings...
        if typeof(e) <: PosDefException
            @warn("You data is collinear. That happens. GLM package is stupid.")
        end
        @warn("GLM had an issue $e")
        return zeros(size(x,1))
    end
end

"Elastic net logistic regression with crossvalidation"
function ENLRCV(x::AbstractMatrix, y::AbstractMatrix, alpha=0., nfolds::Int=min(10, div(size(y, 2), 3)))
    n1 = size(x,2);
    n2 = size(y,2);
    X = copy(hcat(x, y)');
    Yglmnet = zeros(n1+n2, 2);
    @inbounds Yglmnet[1:n1, 1] .= 1;
    @inbounds Yglmnet[n1+1:end, 2] .= 1;

    try
        cv = glmnetcv(X, Yglmnet, Binomial(), alpha=alpha, nfolds=nfolds)
        return GLMNet.coef(cv)
    catch e
        # Convert Errors to Warnings...
        @warn("GLMNet had an issue $e")
        return zeros(size(x,1))
    end
end

"Use half of the data for training, and evaluate CP on the other half"
function half_and_half_CP(x::AbstractMatrix, y::AbstractMatrix, findLinaerMap::Union{Function,Symbol})
    n1 = size(x,2);
    n2 = size(y,2);
    n  = n1 + n2;
    nfolds = 2;
    
    # split outer folds
    classes = vcat(zeros(n1), ones(n2))
    outerCVScheme = MLBase.StratifiedKfold(classes, nfolds)
    
    train_inds = iterate(outerCVScheme)[1]
    @inbounds xidx = train_inds[train_inds .<= n1]
    @inbounds yidx = train_inds[train_inds .> n1] .- n1
    
    # train on train
    w = execFindLinearMap(findLinaerMap, x[:,xidx], y[:,yidx])
    
    test_inds = setdiff(1:n, train_inds)
    @inbounds xidx2 = test_inds[test_inds .<= n1]
    @inbounds yidx2 = test_inds[test_inds .> n1] .- n1

    # project test
    @inbounds xTest = w' * x[:,xidx2];
    @inbounds yTest = w' * y[:,yidx2];
    
    return CP(xTest', yTest')
end

"Divide the data set into nfolds, pool projections on test sets and returns CP on the combined pool"
function pooledCP(x::AbstractMatrix, y::AbstractMatrix, findLinaerMap::Union{Function,Symbol}; nfolds=nfolds::Int=5)
    n1 = size(x,2);
    n2 = size(y,2);
    n  = n1 + n2;
    
    # split outer folds
    classes = vcat(zeros(n1), ones(n2))
    outerCVScheme = MLBase.StratifiedKfold(classes, nfolds)
    xTest = Array{Any}(undef, nfolds)
    yTest = Array{Any}(undef, nfolds)
    @inbounds for (i, train_inds) ∈ enumerate(outerCVScheme)
        test_inds = setdiff(1:n, train_inds)
        # train on train
        @inbounds xidx = train_inds[train_inds .<= n1]
        @inbounds yidx = train_inds[train_inds .> n1] .- n1
        w = execFindLinearMap(findLinaerMap, x[:,xidx], y[:,yidx])
        # project test
        @inbounds xidx2 = test_inds[test_inds .<= n1]
        @inbounds yidx2 = test_inds[test_inds .> n1] .- n1

        @inbounds xTest[i] = w' * x[:,xidx2];
        @inbounds yTest[i] = w' * y[:,yidx2];
    end

    # evaluate CP on pooled test
    return CP(hcat(xTest...)', hcat(yTest...)')
end

"Bias correction using Jackknife. This is slow for large sample sizes."
function jackknifeCP(x::AbstractMatrix, y::AbstractMatrix, findLinaerMap::Union{Function,Symbol})
    n1 = size(x,2);
    n2 = size(y,2);
    
    eCP = CP(x, y, findLinaerMap);

    # leave-one-out estimation
    nLOO = min(n1, n2);
    jkkCP = Vector{Any}(undef, nLOO)
    xidx = 1:n1
    yidx = 1:n2
    for k = 1:nLOO
        @inbounds xLOO = view(x, :, filter(!isequal(k), xidx))
        @inbounds yLOO = view(y, :, filter(!isequal(k), yidx))
        jkkCP[k] = CP(xLOO, yLOO, findLinaerMap)
    end
    vCP = var(jkkCP) * (nLOO - 1)^2 / nLOO # boostrap estimated variance
    bcCP = nLOO * eCP - (nLOO - 1) * mean(jkkCP) # bias corrected CP
    
    return (bcCP, vCP, eCP)
end

end
