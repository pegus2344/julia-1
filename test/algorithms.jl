
@testset "replace! & replace" begin
    a = [1, 2, 3, 1]
    @test replace(iseven, x->2x, a) == [1, 4, 3, 1]
    @test replace!(iseven, x->2x, a) === a
    @test a == [1, 4, 3, 1]
    @test replace(a, 1, 0) == [0, 4, 3, 0]
    @test replace(a, 1, 0, 1) == [0, 4, 3, 1] # 1 replacement only
    @test replace!(a, 1, 2) == [2, 4, 3, 2]

    d = Dict(1=>2, 3=>4)
    @test replace(x->x.first > 2, d, 0=>0) == Dict(1=>2, 0=>0)
    @test replace!(x->x.first > 2, x->(x.first=>2*x.second), d) ==
        Dict(1=>2, 3=>8)
    @test replace(d, 3=>8, 0=>0) == Dict(1=>2, 0=>0)
    @test replace!(d, 3=>8, 2=>2) === d
    @test d == Dict(1=>2, 2=>2)
    @test replace(x->x.second == 2, d, 0=>0, 1) in [Dict(1=>2, 0=>0),
                                                    Dict(2=>2, 0=>0)]

    s = Set([1, 2, 3])
    @test replace(x->x>1, x->2x, s) == Set([1, 4, 6])
    @test replace(x->x>1, x->2x, s, 1) in [Set([1, 4, 3]), Set([1, 2, 6])]
    @test replace(s, 1, 4) == Set([2, 3, 4])
    @test replace!(s, 1, 2) == Set([2, 3])

    @test !(0 in replace([1, 2, 3], 1, 0, 0)) # count=0 --> no replacements
end
