using Test
using myExample

@testset "AutoDiff smoke test" begin
    # --- Definiowanie zmiennych ---
    x = myExample.AutoDiff.Variable(5.0, name="x")
    two = myExample.AutoDiff.Constant(2.0)

    # --- Definiowanie funkcji: y = sin(x^2) ---
    squared = x^two               # x²
    sine = sin(squared)           # sin(x²)

    # --- Topologiczne sortowanie całego grafu ---
    order = myExample.AutoDiff.topological_sort(sine)

    # --- Obliczanie wartości funkcji ---
    y_val = myExample.AutoDiff.forward!(order)

    # --- Backprop: obliczanie gradientów ---
    myExample.AutoDiff.backward!(order)

    # --- Oczekiwane wartości ---
    expected_y = sin(25.0)
    expected_grad = 2 * 5.0 * cos(25.0)  # dy/dx sin(x²) = cos(x²) * 2x 

    # --- Testy ---
    @test isapprox(y_val, expected_y; atol=1e-10)
    @test isapprox(x.gradient, expected_grad; atol=1e-10)
end
