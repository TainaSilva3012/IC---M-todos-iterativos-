### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ d11ec140-9670-11ec-05d6-290873e97830
using LinearAlgebra, BenchmarkTools, Random, ProximalOperators, Plots, DataFrames, LazySets, SparseArrays, Combinatorics, PlutoUI, InteractiveUtils

# ╔═╡ 0650a17f-71a6-46b4-9580-9dc474fae398
md"""
### Projeções
"""

# ╔═╡ 818d8308-21e4-412f-9c09-3ff7a0b18d82
md"""
Projeção de uma matriz A em um conjunto convexo $C_{1}(D) := \{X \in \mathbb{S}^{n} \mid X \text{ é uma matriz de distância com as entradas } x_{ij} = d_{ij} \forall (i,j) \in \Omega\}$
"""

# ╔═╡ 4f3267cf-6d91-4b18-9737-8cc4545249f8
function proj_matrix_convex11(A::UpperTriangular, D::UpperTriangular)
    n, = size(A)
    P = copy(A)
    for i = 1:n
        for j = i:n
            if D[i, j] ≠ 0
                P[i, j] = D[i, j]
            elseif i == j
                P[i, j] = 0
            else
                P[i, j] = max(A[i, j],0)
            end
        end
        P[i, i] = 0
    end
    return P
end

# ╔═╡ 3f9607fa-db13-4995-ae1a-6a75166d93bc
md"""
Projeção de uma matriz A em um conjunto convexo $C_{2} := \{X \in \mathbb{S}^{n-1} \mid \hat{X} \succeq 0 \}$ em que $\hat{X} \succeq 0$ representa o conjunto das matrizes semi-definidas positivas.
"""

# ╔═╡ 6728b195-52d9-4b2d-bbf6-aee907f7a432
function proj_matrix_convex2(A)
    n, = size(A)
    z = ones(n)
    z[n] += sqrt(n)
    Q = I - 2.0 * (z * z') / (z' * z)
    X = Q * (-Symmetric(A)) * Q
    a = X[1:n-1, n]
    α = X[n, n]
    Â = X[1:n-1, 1:n-1]
    Λ, U = eigen(Â)
    U = real.(U)
    Λ = real.(Λ)
    for i = 1:n-1
        Λ[i] = max(Λ[i], 0.0)
    end
    return UpperTriangular(-Q * [U*diagm(Λ)*U' a; a' α] * Q)
end

# ╔═╡ a43ecf91-65ae-4e28-b139-6d10930728c6
reflec_matrix_convex1(X, D) = 2 * proj_matrix_convex11(X, D) - X

# ╔═╡ ca15e79f-deca-4f72-83e4-f9044c04a12c
reflec_matrix_convex2(X) = 2 * proj_matrix_convex2(X) - X

# ╔═╡ 2f1a83c8-5637-497e-8573-e80bca9d71b2
md"""
Projeção de uma matriz A em um conjunto não convexo $C^r_2 := \{X \in \mathbb{S}^{n-1} \mid \hat{X} \succeq 0 $, com $posto(\hat{X}) = r$ \}$.

"""

# ╔═╡ ec2a7a98-d5c8-4663-a188-0ac2ee3d21fa
function proj_matrix_convex2r(A; r=3)
    n, = size(A)
    z = ones(n)
    z[n] += sqrt(n)
    Q = I - 2.0 * (z * z') / (z' * z)
    X = Q * (-Symmetric(A)) * Q
    a = X[1:n-1, n]
    α = X[n, n]
    Â = X[1:n-1, 1:n-1]
    Λ, U = eigen(Â)
    U = real.(U)
    Λ = real.(Λ)
	sort!(Λ)
	Λ₊ = [zeros(n-r-1); max.(Λ[end-r+1:end],0)]
    return UpperTriangular(-Q * [U*diagm(Λ₊)*U' a; a' α] * Q)
end

# ╔═╡ daa6ab95-80cd-489c-8ad5-60c919fe96b6
reflec_matrix_convex2r(X) = 2 * proj_matrix_convex2r(X) - X

# ╔═╡ 5f614a04-792e-42b1-8c37-e0969795da4c
md"""
### Métodos
"""

# ╔═╡ 0a204912-5996-4dc1-924d-69a5cafa211b
function MAP(A, D; itmax = 10000, ε = 1e-8)
    k = 0
    tol = 1.0
    X₀ = copy(A)
    while k <= itmax && tol > ε
        Y = proj_matrix_convex11(X₀, D)
        Z = proj_matrix_convex2r(Y)
        X₀ = Z
        tol = norm(Y - Z)
        k += 2
    end
    return X₀, tol, k
end

# ╔═╡ e3b92773-1bbe-4766-be05-f699a1445bae
function DRM(A, D; itmax = 10000, ε = 1e-8)
    k = 0
    X₀ = copy(A)
    tol = 1.0
    while k <= itmax && tol > ε
        Y = reflec_matrix_convex2r(X₀)
        Z = reflec_matrix_convex1(Y,D)
        X₀ = (X₀ .+ Z) ./ 2
        k += 2
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2r(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ a0eeb447-91ad-41a5-b0c8-677899d1fd5d
function Circuncentro_Rn(A::Matrix)
    S = A[:, 2:end] .- A[:, 1]
    Gram = S'S
    b = 0.5 .* diag(Gram)
    Sol = Gram \ b
    C = A[:, 1] .+ S * Sol
    return C
end

# ╔═╡ 8dc6faf1-8faf-4994-8477-38777d1c27d8
function Circ_3p(x, y, z)
    if x ≈ y && y ≈ z
        return x
    elseif x ≈ y
        return (x .+ z) ./ 2
    elseif x ≈ z
        return (x .+ y) ./ 2
    elseif z ≈ y
        return (x .+ z) ./ 2
    else
        Sᵤ = y .- x
        Sᵥ = z .- x
        norm_Sᵤ = dot(Sᵤ, Sᵤ)
        norm_Sᵥ = dot(Sᵥ, Sᵥ)
        prod = dot(Sᵤ, Sᵥ)
        A = [norm_Sᵤ prod; prod norm_Sᵥ]
        b = [1 / 2 .* norm_Sᵤ; 1 / 2 .* norm_Sᵥ]
        sol = A \ b
        C = x .+ sol[1] .* Sᵤ .+ sol[2] .* Sᵥ
        return C
    end
end

# ╔═╡ ccc29dab-6873-4e8b-a4c2-2329785173fe
function CRM(A, D; itmax = 10000, ε = 1e-8)
    k = 1
    X₀ = proj_matrix_convex11(A,D)
    tol = 1.0
    while k <= itmax && tol > ε
        Y = reflec_matrix_convex2r(X₀)
        Z = reflec_matrix_convex1(Y,D)
        X₀ = Circ_3p(X₀, Y, Z)
        k += 2
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2r(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ 2aebdeea-951f-401d-b7c0-9c880ae19659
function CCRM(A, D; itmax = 10000, ε = 1e-8)
    k = 0
    tol = 1.0
    X₀ = copy(A)
    while k <= itmax && tol > ε
        Y = proj_matrix_convex11(X₀, D)
        Z = proj_matrix_convex2r(Y)
        W = proj_matrix_convex11(Z, D)
        M = (W .+ Z) ./ 2

        X₀ = Circ_3p(M, reflec_matrix_convex1(M, D), reflec_matrix_convex2r(M))
        k += 4
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2r(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ d8fdbe1a-2fe7-4f4c-b218-7a336a0527f8
function CRMc(A, D; itmax = 10000, ε = 1e-8)
    k = 3
    tol = 1.0
    X = copy(A)
	Y1 = proj_matrix_convex11(X, D)
    Z1 = proj_matrix_convex2r(Y1)
    X₀ = proj_matrix_convex11(Z1, D)

  while k <= itmax && tol > ε
        Y = reflec_matrix_convex2r(X₀)
        Z = reflec_matrix_convex1(Y,D)
        X₀ = Circ_3p(X₀, Y, Z)
        k += 2
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2r(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ cdef6215-6444-4e4c-b3ca-e32e9b56800f
md"""
### TESTE
"""

# ╔═╡ 93f3cbf3-2670-4e27-9bc4-e2118023db58
function PCRMc(X, D; itmax=10000, ε= 1e-8)
	 k = 3
    tol = 1.0
    Q = copy(X)
	Y1 = proj_matrix_convex11(Q, D)
    Z1 = proj_matrix_convex2r(Y1)
    X₀ = proj_matrix_convex11(Z1, D)
	
	while k <= itmax && tol > ε
		Y = reflec_matrix_convex2r(X₀)
		Z = reflec_matrix_convex1(Y,D)
		X₀ = Circ_3p(X₀, Y, Z)
		k+=2
		tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2r(X₀))
	end
	return X₀, tol, k
end

# ╔═╡ fec26f5a-2ac8-4910-bbaf-84b9175149c8
md"""
### Gerar Matrizes (MoreWu)
"""

# ╔═╡ 7db2fab2-dc7e-48d0-99d3-76ac1688a3c7
gerarpts(s::Int) = [[α,β,γ] for α in 0:s-1 for β in 0:s-1 for γ in 0:s-1]
# gera vetor em R^3 combinação de 0,1,...,s-1

# ╔═╡ 4ea3e65f-52d8-446b-a5e5-489a8c36f5cd
dist2(x, y) = norm(x - y)^2

# ╔═╡ 4d3326fd-6b4a-4d32-9d20-0c7b07078840
function gerarmde(pts::Vector)

    numel = length(pts)
    M = zeros(numel, numel)
    for i ∈ 1:numel, j ∈ 1:numel
        M[i, j] = dist2(pts[i], pts[j])
    end
    return M
end

# ╔═╡ 86d7f655-e367-4e49-95a5-675b5a43bf63
function mparcial(M::AbstractMatrix, t::Number = 3)
#pega matriz parcial
    n, = size(M)
    A = copy(M)
    for i ∈ 1:n, j ∈ 1:n
        if A[i, j] > t
			A[i, j] = 0
        end
    end
    return A
end

# ╔═╡ 1d228d38-d550-469c-87b8-c043813356a4
begin

sparse(mparcial(gerarmde(gerarpts(3)),1))
end

# ╔═╡ a50a1b0c-5c92-447b-9f8b-16cbc3d27abc
md"""
### Teste Completamento com Tempo $C_1 \cap C_2^r$.
"""

# ╔═╡ f24762e9-c989-4c7d-9c8c-0682c0417c15
function testemorewutemponoconv(s, t; itmax = 5000, ε = 1e-5, tempo = false)
    df = DataFrame(alg = String[], numproj = Int[], tol = Float64[], time = Float64[])
    pts = gerarpts(s)
    M = gerarmde(pts)
    D = UpperTriangular(mparcial(M, t))
    n, = size(D)
    X = UpperTriangular(zeros(n, n))
	tcrm, tccrm, tmap, tdrm, tCRMc, tPCRMc = NaN, NaN, NaN, NaN, NaN, NaN
	
    if tempo
        tcrm = @belapsed CRM($X, $D, itmax = $itmax, ε = $ε)
        tccrm = @belapsed CCRM($X, $D, itmax = $itmax, ε = $ε)
 		tmap = @belapsed MAP($X, $D, itmax = $itmax, ε = $ε)
		tdrm = @belapsed DRM($X, $D, itmax = $itmax, ε = $ε)
		#TESTE
		tCRMc = @belapsed CRMc($X, $D, itmax = $itmax, ε = $ε)
		#tPCRMc = @belapsed PCRMc($X, $D, itmax = $itmax, ε = $ε)
    end
	
    Xcrm, tolcrm, kcrm = CRM(X, D, itmax = itmax, ε = ε)
    push!(df, ["CRM", kcrm, tolcrm, tcrm])
	
    Xccrm, tolccrm, kccrm = CCRM(X, D, itmax = itmax, ε = ε)
    push!(df, ["CCRM", kccrm, tolccrm, tccrm])
	
	Xmap, tolmap, kmap = MAP(X,D, itmax = itmax, ε = ε)
    push!(df,["MAP",kmap,tolmap, tmap])
	
	Xdrm, toldrm, kdrm = DRM(X,D, itmax = itmax, ε = ε)
    push!(df,["DRM",kdrm,toldrm, tdrm])

	#TESTE
	
	XCRMc, tolCRMc, kCRMc = CRMc(X,D, itmax = itmax, ε = ε)
    push!(df,["CRMc", kCRMc, tolCRMc, tCRMc])

	#XPCRMc, tolPCRMc, kPCRMc = PCRMc(X,D, itmax = itmax, ε = ε)
    #push!(df,["PCRMc", kPCRMc, tolPCRMc, tPCRMc])
	
	println("Terminado df$(s)$(t)")
    return df
end

# ╔═╡ c99c88d1-2dfd-4407-a757-c8ecd780241d
df31noconv = testemorewutemponoconv(3,1, itmax = 50000, tempo = true)

# ╔═╡ 4c6d0dec-7b5d-4c2c-9210-f79c8d94ffb3
df33noconv = testemorewutemponoconv(3,3, itmax = 100000, tempo = true)

# ╔═╡ f31e1834-9a7b-4975-aedc-eaebb3e7891e
df34noconv = testemorewutemponoconv(3,4, itmax = 100000, tempo = true)

# ╔═╡ c9b12dd2-2f8b-4861-8b2a-3cdbf0de5333
df37noconv = testemorewutemponoconv(3,7, itmax = 100000, tempo = true)

# ╔═╡ 5e308e6c-512a-458f-bdbb-1edd113c1192
#df42noconv = testemorewutemponoconv(4,4, itmax = 100000, tempo = true)

# ╔═╡ b30bd3ea-bd27-4d40-aab7-038a2a30a1e9
#df52noconv = testemorewutemponoconv(5,4, itmax = 100000, tempo = true)

# ╔═╡ 74559956-072a-4f59-9c70-e735ba2b10f6
#df62noconv = testemorewutemponoconv(6,4, itmax = 100000, tempo = true)

# ╔═╡ 9fd6807a-9abf-46c0-b3e8-472c4aee8a06
#df72noconv = testemorewutemponoconv(7,4, itmax = 100000, tempo = true)

# ╔═╡ f599b393-5285-4f0c-876f-e122e3e13b5a
md"""
### Teste Completamento com Tempo $C_1 \cap C_2$.
"""

# ╔═╡ e2e5cc03-7782-4e38-8e5d-cc305242ff41
function MAPconv(A, D; itmax = 10000, ε = 1e-8)
    k = 0
    tol = 1.0
    X₀ = copy(A)
    while k <= itmax && tol > ε
        Y = proj_matrix_convex11(X₀, D)
        Z = proj_matrix_convex2(Y)
        X₀ = Z
        tol = norm(Y - Z)
        k += 2
    end
    return X₀, tol, k
end

# ╔═╡ c4b2c5ad-f143-4059-abde-8e9c508240c0
function DRMconv(A, D; itmax = 10000, ε = 1e-8)
    k = 0
    X₀ = copy(A)
    tol = 1.0
    while k <= itmax && tol > ε
        Y = reflec_matrix_convex2(X₀)
        Z = reflec_matrix_convex1(Y,D)
        X₀ = (X₀ .+ Z) ./ 2
        k += 2
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ 78606bd2-75c7-41c9-9135-9c2606b866f3
function CRMconv(A, D; itmax = 10000, ε = 1e-8)
    k = 1
    X₀ = proj_matrix_convex11(A,D)
    tol = 1.0
    while k <= itmax && tol > ε
        Y = reflec_matrix_convex2(X₀)
        Z = reflec_matrix_convex1(Y,D)
        X₀ = Circ_3p(X₀, Y, Z)
        k += 2
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ 1a4eafa9-6293-4174-be89-a449cca6779d
function CCRMconv(A, D; itmax = 10000, ε = 1e-8)
    k = 0
    tol = 1.0
    X₀ = copy(A)
    while k <= itmax && tol > ε
        Y = proj_matrix_convex11(X₀, D)
        Z = proj_matrix_convex2(Y)
        W = proj_matrix_convex11(Z, D)
        M = (W .+ Z) ./ 2

        X₀ = Circ_3p(M, reflec_matrix_convex1(M, D), reflec_matrix_convex2(M))
        k += 4
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ 1c203802-4e29-4632-8051-b4fd1d7bc9d6
function CRMcconv(A, D; itmax = 10000, ε = 1e-8)
    k = 3
    tol = 1.0
    X = copy(A)
	Y1 = proj_matrix_convex11(X, D)
    Z1 = proj_matrix_convex2(Y1)
    X₀ = proj_matrix_convex11(Z1, D)

  while k <= itmax && tol > ε
        Y = reflec_matrix_convex2(X₀)
        Z = reflec_matrix_convex1(Y,D)
        X₀ = Circ_3p(X₀, Y, Z)
        k += 2
        tol = norm(proj_matrix_convex11(X₀, D) - proj_matrix_convex2(X₀))
    end
    return X₀, tol, k
end

# ╔═╡ 88200fee-a109-49cd-ac94-b6f34e7b2f65
function testemorewutempoconv(s, t; itmax = 10000, ε = 1e-5, tempo = false)
    df = DataFrame(alg = String[], numproj = Int[], tol = Float64[], time = Float64[])
    pts = gerarpts(s)
    M = gerarmde(pts)
    D = UpperTriangular(mparcial(M, t))
    n, = size(D)
    X = UpperTriangular(zeros(n, n))
	tcrm, tccrm, tmap, tdrm, tCRMc, tPCRMc = NaN, NaN, NaN, NaN, NaN, NaN
	
	
    if tempo
        tcrm = @belapsed CRMconv($X, $D, itmax = $itmax, ε = $ε)
        tccrm = @belapsed CCRMconv($X, $D, itmax = $itmax, ε = $ε)
 		tmap = @belapsed MAPconv($X, $D, itmax = $itmax, ε = $ε)
		tdrm = @belapsed DRMconv($X, $D, itmax = $itmax, ε = $ε)
		#TESTE
		tCRMc = @belapsed CRMcconv($X, $D, itmax = $itmax, ε = $ε)
		#tPCRMc = @belapsed PCRMc($X, $D, itmax = $itmax, ε = $ε)
    end
	
    Xcrm, tolcrm, kcrm = CRMconv(X, D, itmax = itmax, ε = ε)
    push!(df, ["CRM", kcrm, tolcrm, tcrm])
	
    Xccrm, tolccrm, kccrm = CCRMconv(X, D, itmax = itmax, ε = ε)
    push!(df, ["CCRM", kccrm, tolccrm, tccrm])
	
	Xmap, tolmap, kmap = MAPconv(X,D, itmax = itmax, ε = ε)
    push!(df,["MAP",kmap,tolmap, tmap])
	
	Xdrm, toldrm, kdrm = DRMconv(X,D, itmax = itmax, ε = ε)
    push!(df,["DRM",kdrm,toldrm, tdrm])

	#TESTE
	
	XCRMc, tolCRMc, kCRMc = CRMcconv(X,D, itmax = itmax, ε = ε)
    push!(df,["CRMc", kCRMc, tolCRMc, tCRMc])

	#XPCRMc, tolPCRMc, kPCRMc = PCRMc(X,D, itmax = itmax, ε = ε)
    #push!(df,["PCRMc", kPCRMc, tolPCRMc, tPCRMc])
	
	println("Terminado df$(s)$(t)")
    return df
end

# ╔═╡ 3d852b11-b2de-40e9-be97-6eadc1f827d6
df32conv = testemorewutempoconv(3,2, itmax = 100000, tempo = true)

# ╔═╡ 1b3bf2f3-ab80-4ca4-95c9-8731349132f9
df33conv = testemorewutempoconv(3,3, itmax = 100000, tempo = true)

# ╔═╡ ad6f186c-b366-4581-9117-0a691fab2195
df34conv = testemorewutempoconv(3,4, itmax = 100000, tempo = true)

# ╔═╡ 5da2cfe0-6cdd-424e-be33-ac00d7e8c37d
df36conv = testemorewutempoconv(3,6, itmax = 100000, tempo = true)

# ╔═╡ ce05732e-d46c-4c90-b69b-317e15e41239
#df42conv = testemorewutempoconv(4,4, itmax = 50000, tempo = true)

# ╔═╡ 729b1448-5010-484a-951a-620985414cf4
#df52conv = testemorewutempoconv(5,4, itmax = 50000, tempo = true)

# ╔═╡ 3f2c5bf0-3f8a-4701-b688-07566fa58d5a
#df62conv = testemorewutempoconv(6,4, itmax = 50000, tempo = true)

# ╔═╡ bd14d180-8504-4aa4-99ec-eb60c0dd0a60
#df72conv = testemorewutempoconv(7,4, itmax = 50000, tempo = true)

# ╔═╡ 63a4f9f6-fd42-4966-9ae5-d8fc3c395b9e


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProximalOperators = "a725b495-10eb-56fe-b38b-717eba820537"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
BenchmarkTools = "~1.3.1"
Combinatorics = "~1.0.2"
DataFrames = "~1.3.2"
LazySets = "~1.55.0"
Plots = "~1.25.11"
PlutoUI = "~0.7.35"
ProximalOperators = "~0.15.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FastRounding]]
deps = ["ErrorfreeArithmetic", "Test"]
git-tree-sha1 = "224175e213fd4fe112db3eea05d66b308dc2bf6b"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.2.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[GLPK]]
deps = ["BinaryProvider", "CEnum", "GLPK_jll", "Libdl", "MathOptInterface"]
git-tree-sha1 = "6f4e9754ee93e2b2ff40c0b0a6b4cdffd289190d"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "0.15.3"

[[GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "MbedTLS", "Sockets"]
git-tree-sha1 = "c7ec02c4c6a039a98a15f955462cd7aea5df4508"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.8.19"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalArithmetic]]
deps = ["CRlibm", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "bbf2793a70c0a7aaa09aa298b277fe1b90e06d78"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.20.3"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "Random", "SparseArrays", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "fe0f87cc077fc6a23c21e469318993caf2947d10"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "0.22.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a6552bfeab40de157a297d84e03ade4b8177677f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.12"

[[LazySets]]
deps = ["Distributed", "ExprTools", "GLPK", "InteractiveUtils", "IntervalArithmetic", "JuMP", "LinearAlgebra", "Random", "RecipesBase", "Reexport", "Requires", "SharedArrays", "SparseArrays"]
git-tree-sha1 = "e92e22dcd8abf31f9418213936aa41399740dd94"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "1.55.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "e8c9653877adcf8f3e7382985e535bb37b083598"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.10.9"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "842b5ccd156e432f369b204bb704fd4020e383ac"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.3.3"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OSQP]]
deps = ["BinaryProvider", "Libdl", "LinearAlgebra", "MathOptInterface", "OSQP_jll", "SparseArrays"]
git-tree-sha1 = "bbe5fd540709013d6e43c79dea846d39f488c4c3"
uuid = "ab2f91bb-94b4-55e3-9ba0-7f65df51de79"
version = "0.7.0"

[[OSQP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d0f73698c33e04e557980a06d75c2d82e3f0eb49"
uuid = "9c4f68bf-6205-5545-a508-2878b064d984"
version = "0.600.200+0"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5c907bdee5966a9adb8a106807b7c387e51e4d6c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.11"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "85bf3e4bd279e405f91489ce518dedb1e32119cb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.35"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "de893592a221142f3db370f48290e3a2ef39998f"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.4"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProximalCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8b989d98f46f46d345a81fc717054ea04eaaed8e"
uuid = "dc4f5ac2-75d1-4f31-931e-60435d74994b"
version = "0.1.1"

[[ProximalOperators]]
deps = ["IterativeSolvers", "LinearAlgebra", "OSQP", "ProximalCore", "SparseArrays", "SuiteSparse", "TSVD"]
git-tree-sha1 = "5c9cc50b12f51f48c03ebf2b12ed803e742f7656"
uuid = "a725b495-10eb-56fe-b38b-717eba820537"
version = "0.15.0"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "995a812c6f7edea7527bb570f0ac39d0fb15663c"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "74fb527333e72ada2dd9ef77d98e4991fb185f04"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.1"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TSVD]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "61cd1ce64b4ffb69e2d156ff7166a8eb796d699a"
uuid = "9449cd9e-2762-5aa3-a617-5413e99d722e"
version = "0.4.3"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═d11ec140-9670-11ec-05d6-290873e97830
# ╟─0650a17f-71a6-46b4-9580-9dc474fae398
# ╟─818d8308-21e4-412f-9c09-3ff7a0b18d82
# ╟─4f3267cf-6d91-4b18-9737-8cc4545249f8
# ╟─3f9607fa-db13-4995-ae1a-6a75166d93bc
# ╟─6728b195-52d9-4b2d-bbf6-aee907f7a432
# ╟─a43ecf91-65ae-4e28-b139-6d10930728c6
# ╟─ca15e79f-deca-4f72-83e4-f9044c04a12c
# ╟─2f1a83c8-5637-497e-8573-e80bca9d71b2
# ╠═ec2a7a98-d5c8-4663-a188-0ac2ee3d21fa
# ╟─daa6ab95-80cd-489c-8ad5-60c919fe96b6
# ╟─5f614a04-792e-42b1-8c37-e0969795da4c
# ╟─0a204912-5996-4dc1-924d-69a5cafa211b
# ╟─e3b92773-1bbe-4766-be05-f699a1445bae
# ╟─a0eeb447-91ad-41a5-b0c8-677899d1fd5d
# ╟─8dc6faf1-8faf-4994-8477-38777d1c27d8
# ╟─ccc29dab-6873-4e8b-a4c2-2329785173fe
# ╟─2aebdeea-951f-401d-b7c0-9c880ae19659
# ╟─d8fdbe1a-2fe7-4f4c-b218-7a336a0527f8
# ╟─cdef6215-6444-4e4c-b3ca-e32e9b56800f
# ╟─93f3cbf3-2670-4e27-9bc4-e2118023db58
# ╟─fec26f5a-2ac8-4910-bbaf-84b9175149c8
# ╟─7db2fab2-dc7e-48d0-99d3-76ac1688a3c7
# ╟─4ea3e65f-52d8-446b-a5e5-489a8c36f5cd
# ╟─4d3326fd-6b4a-4d32-9d20-0c7b07078840
# ╟─86d7f655-e367-4e49-95a5-675b5a43bf63
# ╠═1d228d38-d550-469c-87b8-c043813356a4
# ╟─a50a1b0c-5c92-447b-9f8b-16cbc3d27abc
# ╠═f24762e9-c989-4c7d-9c8c-0682c0417c15
# ╠═c99c88d1-2dfd-4407-a757-c8ecd780241d
# ╠═4c6d0dec-7b5d-4c2c-9210-f79c8d94ffb3
# ╠═f31e1834-9a7b-4975-aedc-eaebb3e7891e
# ╠═c9b12dd2-2f8b-4861-8b2a-3cdbf0de5333
# ╠═5e308e6c-512a-458f-bdbb-1edd113c1192
# ╠═b30bd3ea-bd27-4d40-aab7-038a2a30a1e9
# ╠═74559956-072a-4f59-9c70-e735ba2b10f6
# ╠═9fd6807a-9abf-46c0-b3e8-472c4aee8a06
# ╟─f599b393-5285-4f0c-876f-e122e3e13b5a
# ╟─e2e5cc03-7782-4e38-8e5d-cc305242ff41
# ╟─c4b2c5ad-f143-4059-abde-8e9c508240c0
# ╟─78606bd2-75c7-41c9-9135-9c2606b866f3
# ╟─1a4eafa9-6293-4174-be89-a449cca6779d
# ╟─1c203802-4e29-4632-8051-b4fd1d7bc9d6
# ╟─88200fee-a109-49cd-ac94-b6f34e7b2f65
# ╠═3d852b11-b2de-40e9-be97-6eadc1f827d6
# ╠═1b3bf2f3-ab80-4ca4-95c9-8731349132f9
# ╠═ad6f186c-b366-4581-9117-0a691fab2195
# ╠═5da2cfe0-6cdd-424e-be33-ac00d7e8c37d
# ╠═ce05732e-d46c-4c90-b69b-317e15e41239
# ╠═729b1448-5010-484a-951a-620985414cf4
# ╠═3f2c5bf0-3f8a-4701-b688-07566fa58d5a
# ╠═bd14d180-8504-4aa4-99ec-eb60c0dd0a60
# ╠═63a4f9f6-fd42-4966-9ae5-d8fc3c395b9e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
