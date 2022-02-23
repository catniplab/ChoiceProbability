using ChoiceProbability
using Test
using Distributions
using Random
using LinearAlgebra

@testset "1D distributions" begin
    @test 0.794 ≈ CP(0., 1., invCP_normal(0.794), 1.)
    @test 0.55 ≈ CP(0., 1., invCP_normal(0.55), 1.)
    @test 0.5 ≈ CP(Normal(), Normal())
    @test 0.5 ≈ CP(zeros(10), zeros(200))
    @test 0.5 ≈ CP(ones(10), ones(200))
    @test 1 ≈ CP(zeros(10), ones(15))
    @test abs(CP(Chisq(3), Chisq(3))-0.5) < 1e-3
    @test abs(CP(Bernoulli(0.5), Bernoulli(0.5)) - 0.5) < 1e-5
    @test CP(Poisson(100), Bernoulli(0)) < 1e-5
    @test CP(Poisson(0.1), Poisson(100)) > (1 - 1e-4)
    @test CP(Poisson(0.1), Normal(20,0.1)) > (1 - 1e-4)
    @test CP(Normal(20,0.1), Poisson(0.1)) < 1e-4
end

@testset "Inverse CP lookup for 1D" begin
    @test invCP_normal(0.5) == 0.
    @test abs(invCP_normal(0.7) - 0.74161) < 1e-4
    @test abs(invCP_search(0.7, Normal(), μ -> Normal(μ, 1), 0, 10) - 0.74161) < 2e-3
    @test abs(CP(Poisson(2), Poisson(invCP_search(0.7, Poisson(2), λ -> Poisson(λ), 1, 20))) - 0.7) < 1e-3
end

@testset "Linear CP" begin
    @test 0.5 ≈ CP(ones(4,10), ones(4,200), zeros(4))
    # draw two normal distributions
    targetCP = 0.7;
    dim = 10;
    rng = MersenneTwister(20220223)
    ruvec = randn(rng, dim);
    normalize!(ruvec); #ruvec = ruvec / sqrt(ruvec' * ruvec);
    mud = invCP_normal(targetCP);
    d1 = MvNormal(zeros(dim), I)
    d2 = MvNormal(mud * ruvec, I)
    n = 2000;
    x1 = rand(rng, d1, n);
    x2 = rand(rng, d2, n);
    @test ChoiceProbability.CP(x1, x2, ruvec) ≈ 0.70131
    @test ChoiceProbability.jackknifeCP(x1, x2, ChoiceProbability.LDA)[1] ≈ 0.7065172586283097
    @test ChoiceProbability.CP(x1, x2, :LDA) ≈ 0.7025790000000001
    @test ChoiceProbability.CP(x1, x2, :LR) ≈ 0.7025622499999999
    @test ChoiceProbability.CP(x1, x2, :ENLRCV) ≈ 0.70256575
end
