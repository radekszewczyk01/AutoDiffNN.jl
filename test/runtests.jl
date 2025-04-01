using myExample
using Test

myExample.operateOnYX(2,2)

@testset "myExample Tests" begin
    
    result = myExample.operateOnYX(2, 3)
    @test result == 6  

    result = myExample.operateOnYX(-1, 4)
    @test result == -4  

    result = myExample.operateOnYX(0, 0)
    @test result == 0  

    # Test z innymi danymi
    result = myExample.operateOnYX(1, 1)
    @test result == 1
end

