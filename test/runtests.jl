using ChoiceProbability
using Test
using Distributions

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

@testset "Linear CP" begin
    @test 0.5 ≈ CP(ones(4,10), ones(4,200), zeros(4))
end
